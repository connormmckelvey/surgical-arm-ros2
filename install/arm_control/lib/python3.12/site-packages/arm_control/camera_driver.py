import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseArray, Pose
from visualization_msgs.msg import Marker

import cv2 as cv
import pyzed.sl as sl
import numpy as np
import open3d as o3d
import sys

# Importing your validated local body tracking utilities
from arm_control.utilities.ZED_bodytracking_34 import (
    setup_body_tracking,
    get_single_body,
    get_arm_points,
    draw_arm_points_and_lines,
)

# click on planes and then press 'S' to stitch and generate the mesh, centroid, and plane boundaries.
# it first tries to do this using alpha shape, and if that fails, it falls back to convex hull. 
# the centroid is calculated as the area-weighted average of triangle centroids.
# press ESC to reset all selections and clear the mesh memory.
class ZedCameraDriverNode(Node):
    def __init__(self):
        super().__init__('zed_camera_driver')
        
        # --- Configurations ---
        self.arm_to_track = "left"  
        self.alpha = 0.08  # Alpha parameter for alpha shape meshing given in meters, think of it like the "radius" of the shape. Lower values yield tighter meshes, higher values yield looser meshes.

        # --- Unified Publishers ---
        self.rviz_pub = self.create_publisher(Marker, '/camera/visualization', 10)
        self.arm_pose_pub = self.create_publisher(PoseArray, 'camera/human_arm_pose', 10)
        self.normalized_hand_pub = self.create_publisher(Point, 'camera/normalized_hand_position', 10)
        
        # --- Memory Arrays & States ---
        self.accumulated_points = []  
        self.plane_chunks = []         
        self.clicked_pixel = None
        self.final_centroid = None  # will be origin
        
        # Fixed Visual Artifact Buffers
        self.centroid_marker = None
        self.mesh_marker_msg = None
        self.planes_marker_msg = None
        self.live_hand_marker = None  # Live updating element
        
        # --- ZED Hardware Interface Configuration ---
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP 
        
        self.get_logger().info("Establishing links with ZED 2i hardware channels...")
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("CRITICAL: Failed to open ZED camera interface!")
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
        
        stitched_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, self.alpha)
        if stitched_mesh.is_empty():
            stitched_mesh, _ = pcd.compute_convex_hull()
        
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
            
        self.final_centroid = centroid_sum / total_area if total_area > 0 else pcd.get_center()

        # Build Marker ID 0: Surface Centroid (Blue Ball)
        self.centroid_marker = Marker()
        self.centroid_marker.header.frame_id = "zed_camera_frame"
        self.centroid_marker.ns = "surface_calibration"
        self.centroid_marker.id = 0  
        self.centroid_marker.type = Marker.SPHERE
        self.centroid_marker.action = Marker.ADD
        self.centroid_marker.pose.position.x = float(self.final_centroid[0])
        self.centroid_marker.pose.position.y = float(self.final_centroid[1])
        self.centroid_marker.pose.position.z = float(self.final_centroid[2])
        self.centroid_marker.pose.orientation.w = 1.0
        self.centroid_marker.scale.x = self.centroid_marker.scale.y = self.centroid_marker.scale.z = 0.04
        self.centroid_marker.color.r, self.centroid_marker.color.g, self.centroid_marker.color.b, self.centroid_marker.color.a = 0.0, 0.8, 1.0, 1.0

        # Build Marker ID 1: Mesh (Blue Surfaces)
        self.mesh_marker_msg = Marker()
        self.mesh_marker_msg.header.frame_id = "zed_camera_frame"
        self.mesh_marker_msg.ns = "surface_calibration"
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
        self.planes_marker_msg.ns = "surface_calibration"
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

        self.publish_rviz_artifacts()

    def send_rviz_delete_markers(self):
        now = self.get_clock().now().to_msg()
        for m_id in [0, 1, 2, 3]:  
            del_msg = Marker()
            del_msg.header.frame_id = "zed_camera_frame"
            del_msg.header.stamp = now
            del_msg.ns = "surface_calibration"
            del_msg.id = m_id
            del_msg.action = Marker.DELETE
            self.rviz_pub.publish(del_msg)
        self.centroid_marker = self.mesh_marker_msg = self.planes_marker_msg = self.live_hand_marker = self.final_centroid = None

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

            # Render overlay green boundary loops *during* calibration selection
            if self.final_centroid is None:
                for chunk in self.plane_chunks:
                    poly_pixels = []
                    for pt in chunk:
                        if pt[1] > 0:
                            poly_pixels.append([int((pt[0]*self.fx)/pt[1] + self.cx), int((-pt[2]*self.fy)/pt[1] + self.cy)])
                    if len(poly_pixels) > 2:
                        cv.fillPoly(frame, [np.array(poly_pixels, dtype=np.int32)], (0, 180, 0))

            # 2. Extract Body Tracker Skeletal Framework (Once Calibration Complete)
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
                        
                        # Stream raw skeleton tracking data to ROS
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
                        
                        # Publish normalized relative coordinates
                        norm_hand_msg = Point(x=float(norm_x), y=float(norm_y), z=float(norm_z))
                        self.normalized_hand_pub.publish(norm_hand_msg)

                        # ==================================================================
                        # NEW VISUALIZATIONS GENERATED HERE (ID 3 and ID 4)
                        # ==================================================================
                        # Build Live Hand Tracking Dot (Marker ID 3 - Bright Magenta)
                        hand_m = Marker()
                        hand_m.header.frame_id = "zed_camera_frame"
                        hand_m.ns = "surface_calibration"
                        hand_m.id = 3
                        hand_m.type = Marker.SPHERE
                        hand_m.action = Marker.ADD
                        hand_m.pose.position.x = float(hd_xyz[0])
                        hand_m.pose.position.y = float(hd_xyz[1])
                        hand_m.pose.position.z = float(hd_xyz[2])
                        hand_m.pose.orientation.w = 1.0
                        hand_m.scale.x = hand_m.scale.y = hand_m.scale.z = 0.04  # 4cm tracking sphere
                        hand_m.color.r, hand_m.color.g, hand_m.color.b, hand_m.color.a = 1.0, 0.0, 1.0, 1.0
                        self.live_hand_marker = hand_m

                        # Build Floating Dynamic Coordinate Text Tag (Marker ID 4)
                        text_m = Marker()
                        text_m.header.frame_id = "zed_camera_frame"
                        text_m.ns = "surface_calibration"
                        text_m.id = 4
                        text_m.type = Marker.TEXT_VIEW_FACING
                        text_m.action = Marker.ADD
                        # Offset the text 8cm above the physical hand location
                        text_m.pose.position.x = float(hd_xyz[0])
                        text_m.pose.position.y = float(hd_xyz[1])
                        text_m.pose.position.z = float(hd_xyz[2]) + 0.08  
                        text_m.pose.orientation.w = 1.0
                        text_m.scale.z = 0.035  # Text character height (3.5cm)
                        text_m.color.r, text_m.color.g, text_m.color.b, text_m.color.a = 1.0, 1.0, 1.0, 1.0  # Opaque White
                        text_m.text = f"Rel: [{norm_x:.2f}, {norm_y:.2f}, {norm_z:.2f}]"
                        self.live_text_marker = text_m
                        # ==================================================================
                        # Publish the live hand marker and text marker
                        now = self.get_clock().now().to_msg()

                        if self.live_hand_marker is not None and self.live_text_marker is not None:
                            self.live_hand_marker.header.stamp = now
                            self.live_text_marker.header.stamp = now
                            self.rviz_pub.publish(self.live_hand_marker)
                            self.rviz_pub.publish(self.live_text_marker)

            cv.imshow(window_name, frame)
            
            key = cv.waitKey(10) & 0xFF
            if key == 27:  # ESC: Reset
                self.accumulated_points = []
                self.plane_chunks = []
                self.send_rviz_delete_markers()
                self.get_logger().info("Mesh memory and visuals cleared.")
            elif key == ord('s'):  # S: Freeze Mesh and engage tracking pipelines
                self.process_and_update_mesh()

        cv.destroyAllWindows()

    def publish_rviz_artifacts(self):
            now = self.get_clock().now().to_msg()

            if (self.centroid_marker is not None and 
                self.mesh_marker_msg is not None and 
                self.planes_marker_msg is not None):
                
                self.centroid_marker.header.stamp = now
                self.mesh_marker_msg.header.stamp = now
                self.planes_marker_msg.header.stamp = now
                
                self.rviz_pub.publish(self.centroid_marker)
                self.rviz_pub.publish(self.mesh_marker_msg)
                self.rviz_pub.publish(self.planes_marker_msg)


    def cleanup(self):
        self.get_logger().info("Safely closing ZED peripheral hardware hooks.")
        self.zed.close()

def main(args=None):
    rclpy.init(args=args)
    node = ZedCameraDriverNode()
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