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
        # Subscribe to color and aligned depth image topics.
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # adjust if your topic is different
            self.image_callback,
            10)
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',  # aligned depth topic
            self.depth_callback,
            10)
        
        self.bridge = CvBridge()
        self.latest_depth = None

        # Load your YOLO model.
        self.model = YOLO('/home/pk/temp_scripts/yolo11n.pt')  # change to your checkpoint's path
        self.get_logger().info("YOLO model loaded and subscribers set up.")

    def depth_callback(self, msg):
        try:
            # Convert the depth ROS image to an OpenCV image.
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV BGR image.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Color image conversion failed: {e}")
            return

        # Perform inference with YOLO.
        results = self.model.predict(cv_image, conf=0.25)

        # Process detection results.
        for result in results:
            boxes = result.boxes
            # Optionally, get the names mapping from your model (if available)
            names = self.model.names if hasattr(self.model, "names") else {}

            for box in boxes:
                # Extract bounding box coordinates (x1, y1, x2, y2).
                coords = box.xyxy[0].cpu().numpy()  # assuming a tensor output; adjust as needed
                x1, y1, x2, y2 = map(int, coords)

                # Compute the centroid of the bounding box.
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Extract depth information at the centroid if a depth frame is available.
                depth_info = "N/A"
                if self.latest_depth is not None:
                    # Ensure the centroid is within the depth image bounds.
                    if 0 <= cx < self.latest_depth.shape[1] and 0 <= cy < self.latest_depth.shape[0]:
                        depth_value = self.latest_depth[cy, cx]
                        # Convert depth value to meters if needed.
                        if self.latest_depth.dtype == np.uint16:
                            # Assuming the value is in millimeters.
                            depth_in_meters = depth_value / 1000.0
                        else:
                            depth_in_meters = depth_value
                        depth_info = f"{depth_in_meters:.2f} m"

                # Extract the detected object's class and confidence if available.
                # Here we assume the detection output provides a tensor for class and confidence.
                # Adjust according to your model's output format.
                if hasattr(box, "cls") and hasattr(box, "conf"):
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = names.get(class_id, "unknown") if names else str(class_id)
                    confidence = box.conf[0].cpu().numpy()
                    label = f"{class_name} {confidence:.2f}, {depth_info}"
                else:
                    # Fallback: just display the depth info.
                    label = depth_info

                # Draw the bounding box on the color image.
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Annotate the bounding box with the object label and depth information.
                cv2.putText(cv_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the annotated image.
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
