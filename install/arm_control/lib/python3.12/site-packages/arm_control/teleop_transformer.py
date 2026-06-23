#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker # Added for 3D visual debugging
import numpy as np

class TeleopTransformerNode(Node):
    def __init__(self):
        super().__init__('teleop_transformer')
        
        # Pulling the movement scaling ratio parameter
        self.declare_parameter('scale_factor', 0.75)       
        self.scale_factor = self.get_parameter('scale_factor').value

        # Placeholder parameter for when you turn the filter back on
        self.declare_parameter('filter_alpha', 0.75) 
        self.filter_alpha = self.get_parameter('filter_alpha').value

        # Baseline "Home" pose matched to your 3-segment extended pose
        self.robot_base_offset = np.array([0.0, 0.0, 0.243]) 
        self.filtered_target = np.copy(self.robot_base_offset)

        # Updated safety clipping bounds
        self.x_bounds = [0.05, 0.42]
        self.y_bounds = [-0.25, 0.25]
        self.z_bounds = [0.02, 0.35]

        # Subscribers and Publishers
        self.arm_pose_sub = self.create_subscription(
            PoseArray, 'camera/human_arm_pose', self.arm_pose_callback, 10)
        
        self.target_pose_pub = self.create_publisher(
            Point, '/arm/target_cartesian_pose', 10)
            
        # Visual debug publisher for RViz
        self.marker_pub = self.create_publisher(
            Marker, '/arm/target_pose_marker', 10)

        self.get_logger().info("Teleop Transformer Online with RViz Visual Marker Streaming.")

    def arm_pose_callback(self, msg):
        if len(msg.poses) < 2:
            return 

        try:
            # Reconstruct numpy arrays from ZED
            shoulder = np.array([msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z])
            wrist = np.array([msg.poses[1].position.x, msg.poses[1].position.y, msg.poses[1].position.z])

            # Extract relative human displacement vector
            human_displacement = wrist - shoulder
            #print(f"Raw human displacement vector: {human_displacement}")

            # Coordinate transformation (Direct 1-to-1 mapping)
            robot_mapped_vector = np.array([
                human_displacement[0],   
                -human_displacement[1],   
                human_displacement[2]    
            ])

            #print(f"Mapped robot displacement vector before scaling: {robot_mapped_vector}")

            # Apply scaling and your custom zero-degree offset
            raw_target = (robot_mapped_vector * self.scale_factor) + self.robot_base_offset

            #print(f"Raw target position before filtering and clipping: {raw_target}")

            # Safety Bounding Box Clipping
            clipped_target = np.array([
                np.clip(raw_target[0], self.x_bounds[0], self.x_bounds[1]),
                np.clip(raw_target[1], self.y_bounds[0], self.y_bounds[1]),
                np.clip(raw_target[2], self.z_bounds[0], self.z_bounds[1])
            ])

            print(f"Clipped target position: {clipped_target}")

            # --- 1. Publish standard Point for the motion planner ---
            cmd_msg = Point()
            cmd_msg.x = float(clipped_target[0])
            cmd_msg.y = float(clipped_target[1])
            cmd_msg.z = float(clipped_target[2])
            self.target_pose_pub.publish(cmd_msg)

            # --- 2. Publish 3D Marker for visual debugging ---
            marker = Marker()
            marker.header.frame_id = "base_link" # Match this to your robot's baseline frame name
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "teleop_target"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position the marker at the clipped target location
            marker.pose.position.x = cmd_msg.x
            marker.pose.position.y = cmd_msg.y
            marker.pose.position.z = cmd_msg.z
            marker.pose.orientation.w = 1.0 # Normalized orientation
            
            # Size of the sphere (0.03 = 3 centimeters wide)
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            # Color configuration (Bright translucent Green)
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8 # Transparency
            
            self.marker_pub.publish(marker)

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
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()