#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO  # Ensure ultralytics is installed

class RealsenseViewer(Node):
    def __init__(self):
        super().__init__('realsense_viewer')
        # Subscribe to color and depth image topics
        self.color_sub = self.create_subscription(
            Image,
            'camera/camera/color/image_raw',
            self.image_callback,
            10)
        self.depth_sub = self.create_subscription(
            Image,
            'camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10)
        
        self.bridge = CvBridge()
        self.latest_depth = None

        # Load YOLOv11n model
        self.model = YOLO('/home/pk/temp_scripts/yolo11n.pt')  # Adjust to your model path
        self.get_logger().info("YOLOv11n model loaded and subscribers initialized.")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Color image conversion failed: {e}")
            return

        results = self.model.predict(cv_image, conf=0.25)

        for result in results:
            boxes = result.boxes
            names = self.model.names  # Class names
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)

                # Compute centroid
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Extract depth
                depth_info = "N/A"
                if self.latest_depth is not None:
                    if 0 <= cx < self.latest_depth.shape[1] and 0 <= cy < self.latest_depth.shape[0]:
                        depth_value = self.latest_depth[cy, cx]
                        if self.latest_depth.dtype == np.uint16:
                            depth_in_meters = depth_value / 1000.0
                        else:
                            depth_in_meters = depth_value
                        depth_info = f"{depth_in_meters:.2f} m"

                # Get label and confidence
                class_id = int(box.cls[0].cpu().numpy())
                class_name = names[class_id] if class_id < len(names) else "unknown"
                confidence = box.conf[0].cpu().numpy()
                label = f"{class_name} {confidence:.2f}, {depth_info}"

                # Draw bounding box and label
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show annotated image
        cv2.imshow("RealSense Color Stream", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = RealsenseViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()