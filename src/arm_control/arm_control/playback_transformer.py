#!/usr/bin/env python3

import os
import json
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker

class PlaybackTransformerNode(Node):
    def __init__(self):
        super().__init__('playback_transformer')

        # Parameters
        self.declare_parameter('calibration_path', '/home/connor/robotics_projects/surgical-arm-ros2/calibration.json')
        calibration_path = self.get_parameter('calibration_path').value

        # Safety Bounding Box bounds
        self.x_bounds = [0.05, 0.42]
        self.y_bounds = [-0.25, 0.25]
        self.z_bounds = [0.02, 0.35]

        # Load Calibration Matrix T_R_C
        self.T_R_C = None
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, 'r') as f:
                    calib_data = json.load(f)
                    self.T_R_C = np.array(calib_data["T_R_C"])
                self.get_logger().info(f"Successfully loaded calibration matrix from {calibration_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load calibration JSON: {e}")
        else:
            self.get_logger().error(f"Calibration file NOT found at {calibration_path}! Please run the eye_to_hand_calibration node first.")

        # State variables
        self.execution_centroid = None

        # Subscribers
        self.centroid_sub = self.create_subscription(
            Point, 'camera/execution_centroid', self.centroid_callback, 10)
        self.norm_hand_sub = self.create_subscription(
            Point, 'camera/normalized_hand_position', self.normalized_hand_callback, 10)

        # Publishers
        self.target_pose_pub = self.create_publisher(
            Pose, '/arm/target_cartesian_pose', 10)
        self.marker_pub = self.create_publisher(
            Marker, '/arm/target_pose_marker', 10)

        self.get_logger().info("Playback Transformer Node online.")

    def centroid_callback(self, msg):
        self.execution_centroid = np.array([msg.x, msg.y, msg.z])
        self.get_logger().info(f"Received execution centroid: [{msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f}]")

    def normalized_hand_callback(self, msg):
        if self.T_R_C is None:
            self.get_logger().error("Cannot transform: Calibration T_R_C matrix is not loaded!", throttle_duration_sec=3.0)
            return

        if self.execution_centroid is None:
            self.get_logger().warn("Centroid not set. Please click on the target object in camera_execution window first.", throttle_duration_sec=3.0)
            return

        # 1. Reconstruct target point in camera frame
        p_centroid = np.array([msg.x, msg.y, msg.z])
        p_camera = self.execution_centroid + p_centroid

        # 2. Transform target point from Camera frame to Robot Base frame using T_R_C
        p_camera_homo = np.append(p_camera, 1.0)
        p_robot_homo = self.T_R_C @ p_camera_homo
        p_robot = p_robot_homo[:3]

        # 3. Apply Safety Bounds clipping
        clipped_target = np.array([
            np.clip(p_robot[0], self.x_bounds[0], self.x_bounds[1]),
            np.clip(p_robot[1], self.y_bounds[0], self.y_bounds[1]),
            np.clip(p_robot[2], self.z_bounds[0], self.z_bounds[1])
        ])

        self.get_logger().info(f"Playback target: X={clipped_target[0]:.3f}, Y={clipped_target[1]:.3f}, Z={clipped_target[2]:.3f}", throttle_duration_sec=2.0)

        # 4. Publish target Pose for motion planner
        cmd_msg = Pose()
        cmd_msg.position.x = float(clipped_target[0])
        cmd_msg.position.y = float(clipped_target[1])
        cmd_msg.position.z = float(clipped_target[2])
        cmd_msg.orientation.x = 0.0
        cmd_msg.orientation.y = 0.0
        cmd_msg.orientation.z = 0.0
        cmd_msg.orientation.w = 1.0
        self.target_pose_pub.publish(cmd_msg)

        # 5. Publish RViz marker
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "playback_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = cmd_msg.position.x
        marker.pose.position.y = cmd_msg.position.y
        marker.pose.position.z = cmd_msg.position.z
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        
        # Bright translucent orange/yellow target sphere
        marker.color.r = 1.0
        marker.color.g = 0.6
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = PlaybackTransformerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
