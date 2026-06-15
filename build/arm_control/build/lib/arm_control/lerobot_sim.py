#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

# Importing your custom robotics utilities for full arm forward kinematics
from arm_control.utilities.fk import space_product_of_exponentials
from arm_control.utilities.se3 import screw_axis_from_w_q

class LeRobotSimulatedDriverNode(Node):
    def __init__(self):
        super().__init__('lerobot_driver') # Kept same name so your ROS network matches
        
        # Exact motion parameters from your physical hardware driver
        self.max_step_deg = 2.0
        self.control_loop_period = 0.05  # 20Hz loop

        # 1. Define Robot Kinematic Configuration (SO-101 Arm) for RViz rendering
        self.w1 = np.array([0, 0, 1])
        self.q1 = np.array([0.038, 0, 0.064])
        self.w2 = np.array([0, 1, 0])
        self.q2 = np.array([0.06874, 0, 0.117050])
        self.w3 = np.array([0, 1, 0])
        self.q3 = np.array([0.097, 0, 0.228])
        self.w4 = np.array([0, 1, 0])
        self.q4 = np.array([0.225, 0, 0.228])
        self.w5 = np.array([1, 0, 0])
        self.q5 = np.array([0.289, 0, 0.228])
        self.w6 = np.array([0, 1, 0])
        self.q6 = np.array([0.314, 0, 0.243])

        self.M = np.array([
            [1, 0, 0, 0.391],
            [0, 1, 0, 0.000],
            [0, 0, 1, 0.243],
            [0, 0, 0, 1.000]
        ])

        self.S_list = [
            screw_axis_from_w_q(self.w1, self.q1),
            screw_axis_from_w_q(self.w2, self.q2),
            screw_axis_from_w_q(self.w3, self.q3),
            screw_axis_from_w_q(self.w4, self.q4),
            screw_axis_from_w_q(self.w5, self.q5),
            screw_axis_from_w_q(self.w6, self.q6),
        ]
        
        # 2. Match your hardware startup behavior (Zero out all joints to Home)
        self.get_logger().info("Simulating Hardware Initialization: Zeroing virtual joints...")
        self.current_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.target_positions = np.copy(self.current_positions)

        # 3. ROS2 Subscribers and Publishers
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/arm/target_joint_angles',
            self.joint_angle_callback,
            10
        )

        self.current_state_pub = self.create_publisher(
            Float32MultiArray, 
            '/arm/current_joint_angles', 
            10
        )

        # NEW: Publisher to send the entire arm geometry to RViz
        self.rviz_arm_pub = self.create_publisher(
            Marker,
            '/arm/simulated_hardware_mesh',
            10
        )

        # 4. Start Runtime Control Loop (Runs your exact 20Hz clock)
        self.control_timer = self.create_timer(self.control_loop_period, self.control_loop_callback)
        self.get_logger().info("Mock LeRobot Driver Online. Connected to virtual visualization pipeline.")

    def joint_angle_callback(self, msg):
        input_angles = np.array(msg.data, dtype=float)
        if len(input_angles) != 6:
            self.get_logger().error(f"Expected 6 joint angles, received: {len(input_angles)}")
            return
        self.target_positions = input_angles

    def control_loop_callback(self):
        # Your exact physical step-limiting logic intact
        diff = self.target_positions - self.current_positions
        max_diff = np.max(np.abs(diff))
        
        if max_diff > 1e-5:
            step_scale = min(1.0, self.max_step_deg / max_diff)
            self.current_positions += diff * step_scale

        # Continually publish state back to your planner node to satisfy the loop
        state_msg = Float32MultiArray()
        state_msg.data = self.current_positions.tolist()
        self.current_state_pub.publish(state_msg)

        # Draw the full physical arm shape in RViz based on current interpolated states
        self.publish_rviz_arm_skeleton()

    def publish_rviz_arm_skeleton(self):
        """ Computes the live 3D location of every joint link and renders them in RViz """
        theta_rad = np.radians(self.current_positions)
        
        # Build coordinate points for every segment connection point
        joints = [np.array([0.0, 0.0, 0.0])] # Shoulder base origin
        joints.append(self.q1)
        
        # Joint 2 location
        T = space_product_of_exponentials(np.eye(4), self.S_list[:1], theta_rad[:1])
        joints.append((T @ np.append(self.q2, 1.0))[:3])
        
        # Joint 3 location
        T = space_product_of_exponentials(np.eye(4), self.S_list[:2], theta_rad[:2])
        joints.append((T @ np.append(self.q3, 1.0))[:3])

        # Joint 4 location
        T = space_product_of_exponentials(np.eye(4), self.S_list[:3], theta_rad[:3])
        joints.append((T @ np.append(self.q4, 1.0))[:3])

        # Joint 5 location
        T = space_product_of_exponentials(np.eye(4), self.S_list[:4], theta_rad[:4])
        joints.append((T @ np.append(self.q5, 1.0))[:3])

        # Joint 6 location
        T = space_product_of_exponentials(np.eye(4), self.S_list[:5], theta_rad[:5])
        joints.append((T @ np.append(self.q6, 1.0))[:3])

        # Gripper / End-Effector Tip (Full Transform Matrix applied to home matrix M)
        T_ee = space_product_of_exponentials(self.M, self.S_list, theta_rad)
        joints.append(T_ee[:3, 3])

        # Packaging everything into a structural LINE_STRIP marker
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "so101_arm_body"
        marker.id = 101
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Visual thickness of the arm links in meters (3.5 cm thick links)
        marker.scale.x = 0.035 
        
        # Translucent solid cyan body so you can distinctively see the arm structure
        marker.color.r = 0.0
        marker.color.g = 0.7
        marker.color.b = 1.0
        marker.color.a = 0.85
        
        # Stream points sequentially to form the arm structure
        for joint_pt in joints:
            p = Point()
            p.x = float(joint_pt[0])
            p.y = float(joint_pt[1])
            p.z = float(joint_pt[2])
            marker.points.append(p)
            
        self.rviz_arm_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotSimulatedDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()