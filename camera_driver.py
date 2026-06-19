#!/usr/bin/env python3

import os
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Point
from visualization_msgs.msg import Marker
import pyzed.sl as sl
import cv2 as cv
import numpy as np
import struct

# Importing utility functions from your workspace layout
from arm_control.utilities.ZED_bodytracking_34 import (
    setup_body_tracking,
    get_single_body,
    get_arm_points,
    draw_arm_points_and_lines,
)

class ZedArmSensorNode(Node):
    def __init__(self):
        super().__init__('zed_arm_sensor')
        
        # Configure tracking target profile ("right" or "left")
        self.arm_to_track = "left"  
        
        # --- ROS 2 Publishers ---
        self.arm_pose_pub = self.create_publisher(PoseArray, 'camera/human_arm_pose', 10) # Publishes 3D skeletal joint positions for the tracked arm
        self.normalized_hand_pose_pub = self.create_publisher(Point, 'camera/normalized_hand_position', 10) # Publishes just the hand position with centroid as orgin
        self.marker_pub = self.create_publisher(Marker, 'camera/surface_markers', 10) # For both hand and plane visualization markers
        
        # Control rate stride (Evaluates consensus plane every 3rd frame ~5Hz)
        self.frame_counter = 0
        self.eval_stride = 3

        #params for consensus sampling
        self.num_samples = 12 # Number of points to sample around the hand for plane estimation
        self.needed_samples = self.num_samples / 3 # Minimum number of valid plane samples required to attempt consensus (e.g., at least 1/3 of the ring must return valid planes)
        self.radius = 40  # pixel radius for sampling ring around the hand point
        self.similarity_threshold = 0.9 # Cosine similarity threshold for normal vector alignment in consensus (1.0 = perfectly parallel)
        
        # ZED camera initialization parameters
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        
        init_params.camera_resolution = sl.RESOLUTION.VGA  
        init_params.camera_fps = 15
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

        self.get_logger().info("Attempting link establishment with ZED 2i hardware channels...")
        
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("CRITICAL: Failed to open ZED camera interface!")
            raise RuntimeError("ZED initialization failed.")
            
        # Positional tracking required for ZED plane algorithms
        self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())
            
        #camera intrinsics for 3D -> 2D projection math
        cam_info = self.zed.get_camera_information()
        cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
        self.fx, self.fy = cam_params.fx, cam_params.fy
        self.cx, self.cy = cam_params.cx, cam_params.cy
        self.img_w = cam_info.camera_configuration.resolution.width
        self.img_h = cam_info.camera_configuration.resolution.height
            
        #body tracking parameters
        self.body_runtime = setup_body_tracking(self.zed)
        self.image = sl.Mat()
        self.bodies = sl.Bodies()
        self.runtime = sl.RuntimeParameters()
        
        self.timer = self.create_timer(0.066, self.process_frame_callback)
        self.get_logger().info("Consensus marker tracking driver node initialized successfully.")

    def process_frame_callback(self):
        self.frame_counter += 1
        try:
            if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
                return
        except Exception as e:
            self.get_logger().warn(f"USB packet drop recovery: {str(e)}")
            return

        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        frame = self.image.get_data()
        if frame.shape[2] == 4:
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        self.zed.retrieve_bodies(self.bodies, self.body_runtime)
        body = get_single_body(self.bodies, mode="closest")

        hand_3d = None

        if body is not None:
            latest_arm_data = get_arm_points(body, arm=self.arm_to_track)
            if latest_arm_data is not None:
                frame = draw_arm_points_and_lines(frame, latest_arm_data)
                
                sh_xyz = latest_arm_data["shoulder_3d"]
                el_xyz = latest_arm_data["elbow_3d"]
                wr_xyz = latest_arm_data["wrist_3d"]
                hand_3d = latest_arm_data["hand_3d"]
                
                # Publish skeletal tracking at 15Hz
                msg = PoseArray()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "zed_camera_frame"
                for joint in [sh_xyz, el_xyz, wr_xyz, hand_3d]:
                    p = Pose()
                    p.position.x, p.position.y, p.position.z = float(joint[0]), float(joint[1]), float(joint[2])
                    msg.poses.append(p)
                self.arm_pose_pub.publish(msg)

        # PROCESS PLANE VIA CONSENSUS SAMPLING (5Hz)
        if hand_3d is not None and (self.frame_counter % self.eval_stride == 0):
            X, Y, Z = hand_3d[0], hand_3d[1], hand_3d[2]
            
            if Y > 0.1:
                u_hand = int((X / Y) * self.fx + self.cx)
                v_hand = int((-Z / Y) * self.fy + self.cy)
                
                # Execute consensus check
                self.evaluate_surface_consensus(u_hand, v_hand, hand_3d)

        cv.imshow("ZED 2i Tracking Feed", frame)
        cv.waitKey(1)

    def evaluate_surface_consensus(self, u_center, v_center, hand_xyz):
        angles = np.linspace(0, 2 * np.pi, self.num_samples, endpoint=False)
        
        sampled_planes = []
        sampled_normals = []
        
        for a in angles:
            u = int(u_center + self.radius * np.cos(a))
            v = int(v_center + self.radius * np.sin(a))
            
            if 0 <= u < self.img_w and 0 <= v < self.img_h:
                local_plane = sl.Plane()
                err = self.zed.find_plane_at_hit((u, v), local_plane)
                if err == sl.ERROR_CODE.SUCCESS:
                    normal = local_plane.get_normal()
                    if np.all(np.isfinite(normal)):
                        sampled_planes.append(local_plane)
                        sampled_normals.append(normal)
                        
        if len(sampled_planes) < self.needed_samples:
            return

        best_consensus_idx = []
        max_votes = 0

        for i in range(len(sampled_normals)):
            current_votes = []
            for j in range(len(sampled_normals)):
                dot_prod = np.dot(sampled_normals[i], sampled_normals[j])
                if abs(dot_prod) > self.similarity_threshold:
                    current_votes.append(j)
            
            if len(current_votes) > max_votes:
                max_votes = len(current_votes)
                best_consensus_idx = current_votes

        if max_votes >= (len(sampled_planes) // 2 + 1):
            winning_plane = sampled_planes[best_consensus_idx[0]]
            bounds = winning_plane.get_bounds()
            
            if len(bounds) > 3:
                pts_2d = np.array([[pt[0], pt[1]] for pt in bounds], dtype=np.float32)
                hull_indices = cv.convexHull(pts_2d, returnPoints=False)
                interpolated_bounds = [bounds[i[0]] for i in hull_indices]
                
                plane_centroid_xyz = self.get_plane_centroid(interpolated_bounds)
                
                # Normalize the hand position relative to the centroid and publish it
                if plane_centroid_xyz is not None:
                    norm_hand_msg = Point()
                    norm_hand_msg.x = float(hand_xyz[0] - plane_centroid_xyz[0])
                    norm_hand_msg.y = float(hand_xyz[1] - plane_centroid_xyz[1])
                    norm_hand_msg.z = float(hand_xyz[2] - plane_centroid_xyz[2])
                    self.normalized_hand_pose_pub.publish(norm_hand_msg)

                self.publish_visualization_markers(interpolated_bounds, hand_xyz, plane_centroid_xyz)

    def publish_visualization_markers(self, bounds_3d, hand_xyz, plane_centroid_xyz):
        now_msg = self.get_clock().now().to_msg()

        # ==============================================================================
        # 1. HAND MARKER (Sphere)
        # ==============================================================================
        hand_marker = Marker()
        hand_marker.header.stamp = now_msg
        hand_marker.header.frame_id = "zed_camera_frame"
        hand_marker.ns = "hand_tracking"
        hand_marker.id = 0
        hand_marker.type = Marker.SPHERE
        hand_marker.action = Marker.ADD
        hand_marker.pose.position.x = float(hand_xyz[0])
        hand_marker.pose.position.y = float(hand_xyz[1])
        hand_marker.pose.position.z = float(hand_xyz[2])
        hand_marker.pose.orientation.w = 1.0
        hand_marker.scale.x = hand_marker.scale.y = hand_marker.scale.z = 0.05
        hand_marker.color.r, hand_marker.color.g, hand_marker.color.b, hand_marker.color.a = 1.0, 0.0, 0.0, 1.0
        self.marker_pub.publish(hand_marker)

        # ==============================================================================
        # 2. SURFACE CENTROID MARKER (Electric Cyan Sphere)
        # ==============================================================================
        if plane_centroid_xyz is not None:
            centroid_marker = Marker()
            centroid_marker.header.stamp = now_msg
            centroid_marker.header.frame_id = "zed_camera_frame"
            centroid_marker.ns = "plane_centroid"
            centroid_marker.id = 2  
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD
            centroid_marker.pose.position.x = float(plane_centroid_xyz[0])
            centroid_marker.pose.position.y = float(plane_centroid_xyz[1])
            centroid_marker.pose.position.z = float(plane_centroid_xyz[2])
            centroid_marker.pose.orientation.w = 1.0
            centroid_marker.scale.x = centroid_marker.scale.y = centroid_marker.scale.z = 0.07
            centroid_marker.color.r, centroid_marker.color.g, centroid_marker.color.b, centroid_marker.color.a = 0.0, 0.8, 1.0, 1.0
            self.marker_pub.publish(centroid_marker)

        # ==============================================================================
        # 3. SOLID COLORED PLANE MARKER (Triangle List)
        # ==============================================================================
        plane_marker = Marker()
        plane_marker.header.stamp = now_msg
        plane_marker.header.frame_id = "zed_camera_frame"
        plane_marker.ns = "surface_plane"
        plane_marker.id = 1
        plane_marker.type = Marker.TRIANGLE_LIST  
        plane_marker.action = Marker.ADD
        plane_marker.pose.orientation.w = 1.0
        plane_marker.scale.x = plane_marker.scale.y = plane_marker.scale.z = 1.0
        plane_marker.color.r, plane_marker.color.g, plane_marker.color.b, plane_marker.color.a = 0.0, 1.0, 0.0, 0.45
        
        if len(bounds_3d) >= 3:
            anchor = plane_centroid_xyz if plane_centroid_xyz is not None else np.mean(bounds_3d, axis=0)
            p_center = Point()
            p_center.x, p_center.y, p_center.z = float(anchor[0]), float(anchor[1]), float(anchor[2])
            
            for i in range(len(bounds_3d)):
                pt1 = bounds_3d[i]
                pt2 = bounds_3d[(i + 1) % len(bounds_3d)]
                
                p1, p2 = Point(), Point()
                p1.x, p1.y, p1.z = float(pt1[0]), float(pt1[1]), float(pt1[2])
                p2.x, p2.y, p2.z = float(pt2[0]), float(pt2[1]), float(pt2[2])
                
                plane_marker.points.append(p_center)
                plane_marker.points.append(p1)
                plane_marker.points.append(p2)
                
            self.marker_pub.publish(plane_marker)

        # ==============================================================================
        # 4. NEW: GENERATIVE BOUNDARY VERTICES VISUALIZER (Points List)
        # ==============================================================================
        if len(bounds_3d) > 0:
            vertex_marker = Marker()
            vertex_marker.header.stamp = now_msg
            vertex_marker.header.frame_id = "zed_camera_frame"
            vertex_marker.ns = "boundary_vertices"
            vertex_marker.id = 3  # Unique ID so it renders concurrently
            vertex_marker.type = Marker.POINTS
            vertex_marker.action = Marker.ADD
            vertex_marker.pose.orientation.w = 1.0
            
            # For Marker.POINTS, x and y scale control the physical size of the dots (2.5cm dots)
            vertex_marker.scale.x = 0.025
            vertex_marker.scale.y = 0.025
            
            # Color configuration (Bright White dots to pop against the green plane)
            vertex_marker.color.r = 1.0
            vertex_marker.color.g = 1.0
            vertex_marker.color.b = 1.0
            vertex_marker.color.a = 1.0  # Opaque
            
            # Push every point from your Convex Hull into the point buffer array
            for pt in bounds_3d:
                p = Point()
                p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])
                vertex_marker.points.append(p)
                
            self.marker_pub.publish(vertex_marker)

    def get_plane_centroid(self, interpolated_bounds):
        if not interpolated_bounds or len(interpolated_bounds) == 0:
            return None
        pts_np = np.array(interpolated_bounds)
        centroid = np.mean(pts_np, axis=0)
        return centroid

    def destroy_node(self):
        self.get_logger().info("Safely severing camera interface connections.")
        try:
            self.zed.close()
            cv.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ZedArmSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()