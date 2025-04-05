#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class RealsenseViewer(Node):
    def __init__(self):
        super().__init__('realsense_viewer')
        self.subscription = self.create_subscription(
            Image,
            'camera/camera/color/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Display the image using OpenCV
        cv2.imshow("RealSense Color Stream", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = RealsenseViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()