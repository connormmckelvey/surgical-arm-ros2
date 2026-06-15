#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Point
import numpy as np

class TeleopTransformerNode(Node):
    def __init__(self):
        super().__init__('teleop_transformer')
        
        # Pulling the movement scaling ratio parameter
        self.declare_parameter('scale_factor', 0.75)       
        self.scale_factor = self.get_parameter('scale_factor').value

        # Placeholder parameter for when you turn the filter back on
        self.declare_parameter('filter_alpha', 0.15) # Adjusted to a stable low-pass value (0.0 - 1.0)
        self.filter_alpha = self.get_parameter('filter_alpha').value

        # Baseline "Home" pose of the robot arm base mapping (Meters)
        self.robot_base_offset = np.array([0.25, 0.0, 0.15]) 
        self.filtered_target = np.copy(self.robot_base_offset)

        # Hard mechanical limits for safety clipping (In Meters relative to SO-101 base frame)
        self.x_bounds = [0.12, 0.36]
        self.y_bounds = [-0.25, 0.25]
        self.z_bounds = [0.02, 0.32]

        # Subscribers and Publishers
        self.arm_pose_sub = self.create_subscription(
            PoseArray, 'camera/human_arm_pose', self.arm_pose_callback, 10)
        
        self.target_pose_pub = self.create_publisher(
            Point, '/arm/target_cartesian_pose', 10)

        self.get_logger().info("Teleop Transformer Node Online (Direct Raw Tracking Model).")

    def arm_pose_callback(self, msg):
        # Quick data integrity check 
        if len(msg.poses) < 2:
            return 

        try:
            # Reconstruct numpy arrays from the incoming ZED metric message payload
            shoulder = np.array([msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z])
            wrist = np.array([msg.poses[1].position.x, msg.poses[1].position.y, msg.poses[1].position.z])

            # Extract relative human displacement vector
            human_displacement = wrist - shoulder

            # Coordinate transformation: Raw 1-to-1 map since ZED matches RIGHT_HANDED_Z_UP 
            # (Robot X = Cam X [Forward], Robot Y = Cam Y [Left], Robot Z = Cam Z [Up])
            robot_mapped_vector = np.array([
                human_displacement[0],   
                human_displacement[1],   
                human_displacement[2]    
            ])

            # Apply spatial scaling factor and home position base offset
            raw_target = (robot_mapped_vector * self.scale_factor) + self.robot_base_offset

            # Safety Bounding Box Guard Matrix
            clipped_target = np.array([
                np.clip(raw_target[0], self.x_bounds[0], self.x_bounds[1]),
                np.clip(raw_target[1], self.y_bounds[0], self.y_bounds[1]),
                np.clip(raw_target[2], self.z_bounds[0], self.z_bounds[1])
            ])

            # --- COMMENTED OUT FILTER FOR NOW ---
            # To re-enable, uncomment the line below and swap out 'clipped_target' for 'self.filtered_target' in cmd_msg
            # self.filtered_target = (self.filter_alpha * clipped_target) + ((1.0 - self.filter_alpha) * self.filtered_target)

            # Package and push out direct raw targets to Motion Planner Node (Using clipped_target directly)
            cmd_msg = Point()
            cmd_msg.x = float(clipped_target[0])
            cmd_msg.y = float(clipped_target[1])
            cmd_msg.z = float(clipped_target[2])
            
            self.target_pose_pub.publish(cmd_msg)

        except Exception as e:
            self.get_logger().error(f"Transformer logic failure: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TeleopTransformerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()