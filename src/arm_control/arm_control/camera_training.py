import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseArray, Pose
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker

import cv2 as cv
import numpy as np
import open3d as o3d
import sys
import time

import arm_control.utilities.rosbag_capture as rb


class CameraTrainingNode(Node):
    def __init__(self):
        super().__init__('camera_training')
        
        # --- Configurations ---
        self.rosbag_enabled = True  
        self.rosbag_folder = "training_bags"

        self.topics_to_record = [
            "camera/human_arm_pose",
            "camera/normalized_hand_position",
            "camera/visualization",
            "force_sensor/data"
        ]   

        # --- Publishers ---
        self.rviz_pub = self.create_publisher(Marker, 'camera/visualization', 10)
        self.normalized_hand_pub = self.create_publisher(Point, 'camera/normalized_hand_position', 10)
        self.plane_req_pub = self.create_publisher(String, 'camera/get_plane/request', 10)

        # --- Subscribers ---
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        
        self.arm_pose_sub = self.create_subscription(
            PoseArray,
            'camera/human_arm_pose',
            self.arm_pose_callback,
            10
        )

        self.plane_res_sub = self.create_subscription(
            String,
            'camera/get_plane/response',
            self.plane_response_callback,
            10
        )

        # --- Automation State Variables ---
        self.bag_process = None  

        # --- Memory Arrays & States ---
        self.accumulated_points = []  
        self.plane_chunks = []         
        self.final_centroid = None  
        self.final_normal_vector = None  
        
        # Camera intrinsics (received from service response)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # Request tracker
        self.last_request_id = None

        # Fixed Visual Artifact Buffers
        self.centroid_marker_msg = None
        self.mesh_marker_msg = None
        self.planes_marker_msg = None
        self.normal_marker_msg = None
        self.live_hand_marker_msg = None
        self.live_text_marker = None
        self.live_hand_marker = None

        # Setup GUI Window
        self.window_name = "ZED 2i Tracking & Calibration"
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self.mouse_callback)
        self.get_logger().info("UI Window initialized. Listening to zed_driver topics (JSON Mode)...")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if self.final_centroid is None:
                self.get_logger().info(f"Screen clicked. Requesting plane at: ({x}, {y})")
                self.request_plane_at_point(x, y)

    def request_plane_at_point(self, x, y):
        req_id = f"req_{time.time_ns()}"
        self.last_request_id = req_id

        req_data = {
            "x": int(x),
            "y": int(y),
            "request_id": req_id
        }

        msg = String(data=json.dumps(req_data))
        self.plane_req_pub.publish(msg)

    def plane_response_callback(self, msg):
        try:
            data = json.loads(msg.data)
            req_id = data.get("request_id")
        except Exception as e:
            self.get_logger().error(f"Failed to parse plane response JSON: {e}")
            return

        if req_id != self.last_request_id:
            # Ignore responses for older clicks
            return

        if data.get("success"):
            self.fx = data["fx"]
            self.fy = data["fy"]
            self.cx = data["cx"]
            self.cy = data["cy"]

            # Append boundary vertices to memory
            current_chunk = []
            for pt in data["boundary_points"]:
                self.accumulated_points.append(pt)
                current_chunk.append(pt)

            if len(current_chunk) > 0:
                self.plane_chunks.append(current_chunk)
                self.get_logger().info(f"Registered new plane segment with {len(current_chunk)} bounds.")
        else:
            self.get_logger().warn("Driver failed to extract plane geometry at clicked coordinates.")

    def process_and_update_mesh(self):
        if len(self.accumulated_points) < 3:
            self.get_logger().warn("Cannot compute surface plane: Not enough points collected yet!")
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

        self.get_logger().info(f"Surface centroid established: Centroid={self.final_centroid}, Normal={self.final_normal_vector}")

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

        self.publish_rviz_artifacts()

    def send_rviz_delete_markers(self):
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

    def image_callback(self, msg):
        # Package raw bytes directly back to OpenCV image array
        if msg.encoding != "bgr8":
            self.get_logger().error(f"Image encoding error: Expected bgr8, received: '{msg.encoding}'")
            return

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3)).copy()

        # Project boundary lines onto live screen feed
        if self.final_centroid is None and self.fx is not None:
            for chunk in self.plane_chunks:
                poly_pixels = []
                for pt in chunk:
                    if pt[1] > 0.01:
                        # 3D coordinates projected via intrinsics to pixel space
                        u = int((pt[0] * self.fx) / pt[1] + self.cx)
                        v = int((-pt[2] * self.fy) / pt[1] + self.cy)
                        poly_pixels.append([u, v])
                if len(poly_pixels) > 2:
                    cv.fillPoly(frame, [np.array(poly_pixels, dtype=np.int32)], (0, 180, 0))

        # Show frame
        cv.imshow(self.window_name, frame)
        
        # User input hooks
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):  
            self.accumulated_points = []
            self.plane_chunks = []
            self.send_rviz_delete_markers()
            self.get_logger().info("Mesh memory and visuals cleared.")
            if self.rosbag_enabled and self.bag_process is not None:
                rb.stop_rosbag_recording(self.bag_process)
                self.bag_process = None
        elif key == ord('s'):
            if self.rosbag_enabled:
                self.bag_process = rb.start_rosbag_recording(self.topics_to_record, self.rosbag_folder)
                time.sleep(1)
            self.process_and_update_mesh()

    def arm_pose_callback(self, msg):
        if self.final_centroid is not None and len(msg.poses) >= 4:
            # Joint index map: [Shoulder=0, Elbow=1, Wrist=2, Hand=3]
            hd_pose = msg.poses[3]
            hd_xyz = [hd_pose.position.x, hd_pose.position.y, hd_pose.position.z]

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
            rb.stop_rosbag_recording(self.bag_process)
        cv.destroyAllWindows()
        self.get_logger().info("UI Window safely terminated.")


def main(args=None):
    rclpy.init(args=args)
    node = CameraTrainingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()