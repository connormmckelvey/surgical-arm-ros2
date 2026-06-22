import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseArray, Pose
from visualization_msgs.msg import Marker

import cv2 as cv
import pyzed.sl as sl
import numpy as np
import open3d as o3d
import sys
import subprocess
import signal
from datetime import datetime
import time

from arm_control.utilities.ZED_bodytracking_34 import (
    setup_body_tracking,
    get_single_body,
    get_arm_points,
    draw_arm_points_and_lines,
)

class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration')
        
        # --- Configurations ---
        self.arm_to_track = "left"  
        self.alpha = 0.08
        self.rosbag_enabled = True  
        self.rosbag_folder = "training_bags"

        # --- Publishers ---
        self.rviz_pub = self.create_publisher(Marker, 'camera/visualization', 10)
        self.arm_pose_pub = self.create_publisher(PoseArray, 'camera/human_arm_pose', 10)
        self.normalized_hand_pub = self.create_publisher(Point, 'camera/normalized_hand_position', 10)

        # --- Automation State Variables ---
        self.bag_process = None  

        # --- Memory Arrays & States ---
        self.accumulated_points = []  
        self.plane_chunks = []         
        self.clicked_pixel = None
        self.final_centroid = None  
        self.final_normal_vector = None  
        
        # Fixed Visual Artifact Buffers
        self.centroid_marker_msg = None
        self.mesh_marker_msg = None
        self.planes_marker_msg = None
        self.normal_marker_msg = None
        self.live_hand_marker_msg = None
        
        # --- ZED Hardware Interface Configuration ---
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA 
        init_params.camera_fps = 15
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP 
        
        self.get_logger().info("Establishing links with ZED 2i hardware channels...")
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("Failed to open ZED camera interface!")
            sys.exit(1)

        self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        
        # Extract Intrinsics
        cam_info = self.zed.get_camera_information()
        cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
        self.fx, self.fy = cam_params.fx, cam_params.fy
        self.cx, self.cy = cam_params.cx, cam_params.cy
        
        # Body Tracking Setup
        self.body_runtime = setup_body_tracking(self.zed)
        self.image = sl.Mat()
        self.bodies = sl.Bodies()
        self.runtime = sl.RuntimeParameters()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.clicked_pixel = [x, y]

    def process_and_update_mesh(self):
        if len(self.accumulated_points) < 6:
            self.get_logger().warning("Select more areas before stitching.")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.accumulated_points))
        
        stitched_mesh, original_point_cloud = pcd.compute_convex_hull()
        
        vertices = np.asarray(stitched_mesh.vertices)
        triangles = np.asarray(stitched_mesh.triangles)
        total_area = 0.0
        centroid_sum = np.zeros(3)
        
        for tri in triangles:
            p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            tri_center = (p0 + p1 + p2) / 3.0
            tri_area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            total_area += tri_area
            centroid_sum += tri_center * tri_area
            
        self.final_normal_vector = np.asarray(stitched_mesh.compute_triangle_normals().triangle_normals).mean(axis=0)
        self.final_centroid = centroid_sum / total_area if total_area > 0 else pcd.get_center()

        # Build Marker ID 0: Surface Centroid (Blue Ball)
        self.centroid_marker_msg = Marker()
        self.centroid_marker_msg.header.frame_id = "zed_camera_frame"
        self.centroid_marker_msg.ns = "surface_centroid"
        self.centroid_marker_msg.id = 0  
        self.centroid_marker_msg.type = Marker.SPHERE
        self.centroid_marker_msg.action = Marker.ADD
        self.centroid_marker_msg.pose.position.x = float(self.final_centroid[0])
        self.centroid_marker_msg.pose.position.y = float(self.final_centroid[1])
        self.centroid_marker_msg.pose.position.z = float(self.final_centroid[2])
        self.centroid_marker_msg.pose.orientation.w = 1.0
        self.centroid_marker_msg.scale.x = self.centroid_marker_msg.scale.y = self.centroid_marker_msg.scale.z = 0.04
        self.centroid_marker_msg.color.r, self.centroid_marker_msg.color.g, self.centroid_marker_msg.color.b, self.centroid_marker_msg.color.a = 0.0, 0.8, 1.0, 1.0

        # Build Marker ID 1: Mesh (Blue Surfaces)
        self.mesh_marker_msg = Marker()
        self.mesh_marker_msg.header.frame_id = "zed_camera_frame"
        self.mesh_marker_msg.ns = "surface_mesh"
        self.mesh_marker_msg.id = 1  
        self.mesh_marker_msg.type = Marker.TRIANGLE_LIST
        self.mesh_marker_msg.action = Marker.ADD
        self.mesh_marker_msg.pose.orientation.w = 1.0
        self.mesh_marker_msg.scale.x = self.mesh_marker_msg.scale.y = self.mesh_marker_msg.scale.z = 1.0
        self.mesh_marker_msg.color.r, self.mesh_marker_msg.color.g, self.mesh_marker_msg.color.b, self.mesh_marker_msg.color.a = 0.2, 0.6, 1.0, 0.5 
        for tri in triangles:
            for v_idx in tri:
                v = vertices[v_idx]
                self.mesh_marker_msg.points.append(Point(x=float(v[0]), y=float(v[1]), z=float(v[2])))

        # Build Marker ID 2: Wireframe Loops (Green Outlines)
        self.planes_marker_msg = Marker()
        self.planes_marker_msg.header.frame_id = "zed_camera_frame"
        self.planes_marker_msg.ns = "surface_outlines"
        self.planes_marker_msg.id = 2  
        self.planes_marker_msg.type = Marker.LINE_LIST
        self.planes_marker_msg.action = Marker.ADD
        self.planes_marker_msg.pose.orientation.w = 1.0
        self.planes_marker_msg.scale.x = 0.004  
        self.planes_marker_msg.color.g, self.planes_marker_msg.color.a = 1.0, 1.0
        for chunk in self.plane_chunks:
            n = len(chunk)
            for i in range(n):
                p_start, p_end = chunk[i], chunk[(i + 1) % n]
                self.planes_marker_msg.points.append(Point(x=float(p_start[0]), y=float(p_start[1]), z=float(p_start[2])))
                self.planes_marker_msg.points.append(Point(x=float(p_end[0]), y=float(p_end[1]), z=float(p_end[2])))

        # Build Marker ID 5: Surface Normal Vector (Vibrant Orange Arrow)
        self.normal_marker_msg = Marker()
        self.normal_marker_msg.header.frame_id = "zed_camera_frame"
        self.normal_marker_msg.ns = "surface_normal"
        self.normal_marker_msg.id = 5  
        self.normal_marker_msg.type = Marker.ARROW
        self.normal_marker_msg.action = Marker.ADD
        self.normal_marker_msg.pose.orientation.w = 1.0
        
        self.normal_marker_msg.scale.x = 0.015  
        self.normal_marker_msg.scale.y = 0.03   
        self.normal_marker_msg.scale.z = 0.05   
        
        self.normal_marker_msg.color.r = 1.0
        self.normal_marker_msg.color.g = 0.5
        self.normal_marker_msg.color.b = 0.0
        self.normal_marker_msg.color.a = 1.0

        vec_norm = np.linalg.norm(self.final_normal_vector)
        unit_normal = self.final_normal_vector / vec_norm if vec_norm > 0 else np.array([0.0, 0.0, 1.0])
        arrow_length = 0.25  
        
        start_pt = Point(x=float(self.final_centroid[0]), y=float(self.final_centroid[1]), z=float(self.final_centroid[2]))
        end_pt = Point(
            x=float(self.final_centroid[0] + unit_normal[0] * arrow_length),
            y=float(self.final_centroid[1] + unit_normal[1] * arrow_length),
            z=float(self.final_centroid[2] + unit_normal[2] * arrow_length)
        )
        
        self.normal_marker_msg.points.append(start_pt)
        self.normal_marker_msg.points.append(end_pt)



        # ==================================================================
        # AUTOMATION: START ROSBAG RECORDING (With your added topic)
        # ==================================================================
        if self.bag_process is None and self.rosbag_enabled:

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bag_name = f"training_episode_{timestamp}"
            
            topics_to_record = [
                "camera/human_arm_pose",
                "camera/normalized_hand_position",
                "camera/visualization",
                "/training_sensor/data"
            ]
            
            cmd = ["ros2", "bag", "record", "-s", "mcap", "-o", self.rosbag_folder] + topics_to_record

            time.sleep(1)

            self.get_logger().info(f"rosbag started: {bag_name}")
            self.bag_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.publish_rviz_artifacts()

    def send_rviz_delete_markers(self):
        # ==================================================================
        # STOP ROSBAG
        # ==================================================================
        if self.bag_process is not None:
            self.bag_process.send_signal(signal.SIGINT) 
            self.bag_process.wait()                      
            self.bag_process = None
            self.get_logger().info("Rosbag folder successfully generated and closed.")

        now = self.get_clock().now().to_msg()
        marker_ns_list = ["surface_centroid", "surface_mesh", "surface_outlines", "hand_position", "hand_coordinates", "surface_normal"]
        for m_id in [0, 1, 2, 3, 4, 5]:  
            del_msg = Marker()
            del_msg.header.frame_id = "zed_camera_frame"
            del_msg.header.stamp = now
            del_msg.ns = marker_ns_list[m_id]
            del_msg.id = m_id
            del_msg.action = Marker.DELETE
            self.rviz_pub.publish(del_msg)
        self.centroid_marker_msg = self.mesh_marker_msg = self.planes_marker_msg = self.live_hand_marker_msg = self.final_centroid = self.normal_marker_msg = None

    def start_processing_loop(self):
        window_name = "ZED 2i Tracking & Calibration"
        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, self.mouse_callback)

        while rclpy.ok():
            if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            frame = self.image.get_data()
            if frame.shape[2] == 4:
                frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

            # 1. Calibration Clicking Engine
            if self.clicked_pixel is not None and self.final_centroid is None:
                plane = sl.Plane()
                if self.zed.find_plane_at_hit(self.clicked_pixel, plane) == sl.ERROR_CODE.SUCCESS:
                    bounds = plane.get_bounds()
                    current_chunk = []
                    for pt in bounds:
                        self.accumulated_points.append([pt[0], pt[1], pt[2]])
                        current_chunk.append([pt[0], pt[1], pt[2]])
                    if len(current_chunk) > 0:
                        self.plane_chunks.append(current_chunk)
                self.clicked_pixel = None

            if self.final_centroid is None:
                for chunk in self.plane_chunks:
                    poly_pixels = []
                    for pt in chunk:
                        if pt[1] > 0:
                            poly_pixels.append([int((pt[0]*self.fx)/pt[1] + self.cx), int((-pt[2]*self.fy)/pt[1] + self.cy)])
                    if len(poly_pixels) > 2:
                        cv.fillPoly(frame, [np.array(poly_pixels, dtype=np.int32)], (0, 180, 0))

            # 2. Extract Body Tracker Skeletal Framework
            if self.final_centroid is not None:
                self.zed.retrieve_bodies(self.bodies, self.body_runtime)
                body = get_single_body(self.bodies, mode="closest")

                if body is not None:
                    latest_arm_data = get_arm_points(body, arm=self.arm_to_track)
                    if latest_arm_data is not None:
                        frame = draw_arm_points_and_lines(frame, latest_arm_data)
                        
                        sh_xyz = latest_arm_data["shoulder_3d"]
                        el_xyz = latest_arm_data["elbow_3d"]
                        wr_xyz = latest_arm_data["wrist_3d"]
                        hd_xyz = latest_arm_data["hand_3d"]
                        
                        pose_msg = PoseArray()
                        pose_msg.header.stamp = self.get_clock().now().to_msg()
                        pose_msg.header.frame_id = "zed_camera_frame"
                        for joint in [sh_xyz, el_xyz, wr_xyz, hd_xyz]:
                            p = Pose()
                            p.position.x, p.position.y, p.position.z = float(joint[0]), float(joint[1]), float(joint[2])
                            pose_msg.poses.append(p)
                        self.arm_pose_pub.publish(pose_msg)

                        # --- COORD TRANSFORM SUBTRACTION ---
                        norm_x = hd_xyz[0] - self.final_centroid[0]
                        norm_y = hd_xyz[1] - self.final_centroid[1]
                        norm_z = hd_xyz[2] - self.final_centroid[2]
                        
                        norm_hand_msg = Point(x=float(norm_x), y=float(norm_y), z=float(norm_z))
                        self.normalized_hand_pub.publish(norm_hand_msg)

                        # Build Live Hand Tracking Dot
                        hand_m = Marker()
                        hand_m.header.frame_id = "zed_camera_frame"
                        hand_m.ns = "hand_position"
                        hand_m.id = 3
                        hand_m.type = Marker.SPHERE
                        hand_m.action = Marker.ADD
                        hand_m.pose.position.x = float(hd_xyz[0])
                        hand_m.pose.position.y = float(hd_xyz[1])
                        hand_m.pose.position.z = float(hd_xyz[2])
                        hand_m.pose.orientation.w = 1.0
                        hand_m.scale.x = hand_m.scale.y = hand_m.scale.z = 0.04  
                        hand_m.color.r, hand_m.color.g, hand_m.color.b, hand_m.color.a = 1.0, 0.0, 1.0, 1.0
                        self.live_hand_marker = hand_m

                        # Build Floating Dynamic Coordinate Text Tag
                        text_m = Marker()
                        text_m.header.frame_id = "zed_camera_frame"
                        text_m.ns = "hand_coordinates"
                        text_m.id = 4
                        text_m.type = Marker.TEXT_VIEW_FACING
                        text_m.action = Marker.ADD
                        text_m.pose.position.x = float(hd_xyz[0])
                        text_m.pose.position.y = float(hd_xyz[1])
                        text_m.pose.position.z = float(hd_xyz[2]) + 0.08  
                        text_m.pose.orientation.w = 1.0
                        text_m.scale.z = 0.035  
                        text_m.color.r, text_m.color.g, text_m.color.b, text_m.color.a = 1.0, 1.0, 1.0, 1.0  
                        text_m.text = f"Rel: [{norm_x:.2f}, {norm_y:.2f}, {norm_z:.2f}]"
                        self.live_text_marker = text_m
                        
                        now = self.get_clock().now().to_msg()
                        if self.live_hand_marker is not None and self.live_text_marker is not None:
                            self.live_hand_marker.header.stamp = now
                            self.live_text_marker.header.stamp = now
                            self.rviz_pub.publish(self.live_hand_marker)
                            self.rviz_pub.publish(self.live_text_marker)

            cv.imshow(window_name, frame)
            
            key = cv.waitKey(10) & 0xFF
            if key == 27:  
                self.accumulated_points = []
                self.plane_chunks = []
                self.send_rviz_delete_markers()
                self.get_logger().info("Mesh memory and visuals cleared.")
            elif key == ord('s'):  
                self.process_and_update_mesh()

        cv.destroyAllWindows()

    def publish_rviz_artifacts(self):
        now = self.get_clock().now().to_msg()
        if (self.centroid_marker_msg is not None and 
            self.mesh_marker_msg is not None and 
            self.normal_marker_msg is not None and
            self.planes_marker_msg is not None):
            
            self.centroid_marker_msg.header.stamp = now
            self.mesh_marker_msg.header.stamp = now
            self.planes_marker_msg.header.stamp = now
            self.normal_marker_msg.header.stamp = now
            
            self.rviz_pub.publish(self.centroid_marker_msg)
            self.rviz_pub.publish(self.mesh_marker_msg)
            self.rviz_pub.publish(self.planes_marker_msg)
            self.rviz_pub.publish(self.normal_marker_msg)

    def cleanup(self):
        if self.bag_process is not None:
            self.bag_process.send_signal(signal.SIGINT)
            self.bag_process.wait()
        self.get_logger().info("Safely closing ZED peripheral hardware hooks.")
        self.zed.close()

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    try:
        node.start_processing_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()