#!/usr/bin/env python3

import sys
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import String
import pyzed.sl as sl
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot

# Importing utility functions from the workspace
from arm_control.utilities.ZED_bodytracking_34 import (
    setup_body_tracking,
    get_single_body,
    get_arm_points,
    draw_arm_points_and_lines,
)


class ZedDriverNode(Node):
    def __init__(self):
        super().__init__('zed_driver')

        # --- Parameters ---
        self.declare_parameter('arm_to_track', 'left')
        self.arm_to_track = self.get_parameter('arm_to_track').value

        self.declare_parameter('show_visualization', False)
        self.show_visualization = self.get_parameter('show_visualization').value

        self.declare_parameter('camera_fps', 15)
        camera_fps = self.get_parameter('camera_fps').value

        # --- Publishers ---
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.arm_pose_pub = self.create_publisher(PoseArray, 'camera/human_arm_pose', 10)
        self.tag_pose_pub = self.create_publisher(PoseStamped, 'camera/tag_pose', 10)
        self.plane_res_pub = self.create_publisher(String, 'camera/get_plane/response', 10)

        # --- Subscribers ---
        self.plane_req_sub = self.create_subscription(
            String,
            'camera/get_plane/request',
            self.plane_request_callback,
            10
        )

        # --- ZED Hardware Interface Configuration ---
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA
        init_params.camera_fps = camera_fps
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

        self.get_logger().info("Initializing ZED 2i camera driver...")
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("CRITICAL: Failed to open ZED camera hardware interface!")
            sys.exit(1)

        # Enable positional tracking (required for ZED plane detection and spatial features)
        self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())

        # Extract Intrinsics
        cam_info = self.zed.get_camera_information()
        cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
        self.fx, self.fy = cam_params.fx, cam_params.fy
        self.cx, self.cy = cam_params.cx, cam_params.cy

        # Camera Intrinsic Matrix K
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Since ZED SDK outputs pre-rectified left images, lens distortion is already resolved.
        # Setting distortion to 0 prevents double-correction errors.
        self.dist = np.zeros((5, 1), dtype=np.float64)

        # --- ArUco Board Configuration ---
        self.aruco = cv.aruco
        self.dictionary = self.aruco.getPredefinedDictionary(self.aruco.DICT_5X5_100)
        self.detector_params = self.aruco.DetectorParameters()
        self.detector = self.aruco.ArucoDetector(self.dictionary, self.detector_params)

        # Printed target specifications:
        # squares_x = 8, squares_y = 8
        # square size = 10 mm (0.010 m), marker size = 7 mm (0.007 m)
        squares_x = 8
        squares_y = 8
        square_length = 0.010
        marker_length = 0.007

        self.board = self.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            self.dictionary,
        )

        # Body Tracking Setup
        self.body_runtime = setup_body_tracking(self.zed)
        self.image_mat = sl.Mat()
        self.bodies = sl.Bodies()
        self.runtime = sl.RuntimeParameters()

        # Timer to grab frames at matching camera framerate
        timer_period = 1.0 / camera_fps
        self.timer = self.create_timer(timer_period, self.process_frame)

        self.get_logger().info("ZED Driver Node (with ArUco tracking) initialized successfully.")

    def process_frame(self):
        # Grab frame from ZED SDK
        if self.zed.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return

        # Retrieve image frame
        self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        frame = self.image_mat.get_data()

        # Convert BGRA to BGR
        if frame.shape[2] == 4:
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        # --------------------------------------------
        # ArUco Board Tracking & Pose Estimation
        # --------------------------------------------
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        marker_corners, marker_ids, rejected = self.detector.detectMarkers(gray)

        if marker_ids is not None and len(marker_ids) > 0:
            num_detected = len(marker_ids)
            self.get_logger().info(f"ArUco Detection: Found {num_detected} markers in frame.", throttle_duration_sec=2.0)

            # Draw detected marker boundaries and IDs
            self.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

            # Estimate board pose relative to camera
            success, rvec, tvec = self.estimate_board_pose_from_aruco_markers(
                self.board,
                marker_corners,
                marker_ids,
                self.K,
                self.dist,
            )

            if success:
                self.get_logger().info("ArUco Board Pose estimated successfully. Publishing pose...", throttle_duration_sec=2.0)
                # Draw coordinate axes at board origin (length: 3 cm)
                cv.drawFrameAxes(frame, self.K, self.dist, rvec, tvec, 0.03)

                # Convert rotation vector to 3x3 matrix
                R_mat, _ = cv.Rodrigues(rvec)

                # Convert to quaternion rotation components
                r_scipy = ScipyRot.from_matrix(R_mat)
                quat = r_scipy.as_quat()  # [qx, qy, qz, qw]

                # Publish pose relative to optical camera frame
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "zed_camera_optical_frame"

                # Translation vector components
                pose_msg.pose.position.x = float(tvec[0][0])
                pose_msg.pose.position.y = float(tvec[1][0])
                pose_msg.pose.position.z = float(tvec[2][0])

                # Quaternion orientation
                pose_msg.pose.orientation.x = float(quat[0])
                pose_msg.pose.orientation.y = float(quat[1])
                pose_msg.pose.orientation.z = float(quat[2])
                pose_msg.pose.orientation.w = float(quat[3])

                self.tag_pose_pub.publish(pose_msg)

        # Retrieve body skeleton positions
        self.zed.retrieve_bodies(self.bodies, self.body_runtime)
        body = get_single_body(self.bodies, mode="closest")

        if body is not None:
            latest_arm_data = get_arm_points(body, arm=self.arm_to_track)
            if latest_arm_data is not None:
                # Optionally draw skeletal lines on the frame
                frame = draw_arm_points_and_lines(frame, latest_arm_data)

                # Extract joint coordinate vectors
                sh_xyz = latest_arm_data["shoulder_3d"]
                el_xyz = latest_arm_data["elbow_3d"]
                wr_xyz = latest_arm_data["wrist_3d"]
                hd_xyz = latest_arm_data["hand_3d"]

                # Publish skeletal tracking joint array
                pose_msg = PoseArray()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "zed_camera_frame"
                for joint in [sh_xyz, el_xyz, wr_xyz, hd_xyz]:
                    p = Pose()
                    p.position.x = float(joint[0])
                    p.position.y = float(joint[1])
                    p.position.z = float(joint[2])
                    pose_msg.poses.append(p)
                self.arm_pose_pub.publish(pose_msg)

        # Publish raw/annotated image over ROS 2 topic
        self.publish_image(frame)

        # Local debug visualizer (optional)
        if self.show_visualization:
            cv.imshow("ZED Driver Debug Feed", frame)
            cv.waitKey(1)

    def estimate_board_pose_from_aruco_markers(self, board, marker_corners, marker_ids, K, dist):
        """ Matches detected marker IDs to board object points and runs solvePnP """
        if marker_ids is None or len(marker_ids) < 4:
            return False, None, None

        board_ids = board.getIds().flatten()
        board_obj_points = board.getObjPoints()

        obj_points = []
        img_points = []

        for detected_idx, detected_id in enumerate(marker_ids.flatten()):
            matches = np.where(board_ids == detected_id)[0]
            if len(matches) == 0:
                continue

            board_idx = matches[0]

            # 3D corners in board frame
            obj_corners = np.asarray(board_obj_points[board_idx], dtype=np.float32).reshape(4, 3)
            # 2D corners in image frame
            img_corners = np.asarray(marker_corners[detected_idx], dtype=np.float32).reshape(4, 2)

            obj_points.append(obj_corners)
            img_points.append(img_corners)

        if len(obj_points) < 4:
            return False, None, None

        obj_points = np.vstack(obj_points).astype(np.float32)
        img_points = np.vstack(img_points).astype(np.float32)

        success, rvec, tvec = cv.solvePnP(
            obj_points,
            img_points,
            K,
            dist,
            flags=cv.SOLVEPNP_ITERATIVE,
        )

        return success, rvec, tvec

    def publish_image(self, cv_image):
        """ Manually packages numpy image arrays to standard sensor_msgs/Image """
        try:
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "zed_camera_frame"
            msg.height = cv_image.shape[0]
            msg.width = cv_image.shape[1]
            msg.encoding = "bgr8"
            msg.is_bigendian = 0
            msg.step = cv_image.shape[1] * cv_image.shape[2]
            msg.data = cv_image.tobytes()
            self.image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish image frame: {e}")

    def plane_request_callback(self, msg):
        try:
            data = json.loads(msg.data)
            x = int(data["x"])
            y = int(data["y"])
            req_id = data["request_id"]
        except Exception as e:
            self.get_logger().error(f"Failed to parse plane request JSON: {e}")
            return

        self.get_logger().info(f"Received plane request at pixel ({x}, {y}) [ID: {req_id}]")
        
        response_data = {
            "request_id": req_id,
            "success": False
        }
        
        plane = sl.Plane()
        err = self.zed.find_plane_at_hit((x, y), plane)
        
        if err == sl.ERROR_CODE.SUCCESS:
            centroid = plane.get_center()
            normal = plane.get_normal()
            bounds = plane.get_bounds()

            # Verify extraction integrity
            if np.all(np.isfinite(centroid)) and np.all(np.isfinite(normal)):
                response_data["success"] = True
                response_data["centroid"] = [float(centroid[0]), float(centroid[1]), float(centroid[2])]
                response_data["normal"] = [float(normal[0]), float(normal[1]), float(normal[2])]
                response_data["boundary_points"] = [[float(pt[0]), float(pt[1]), float(pt[2])] for pt in bounds]
                response_data["fx"] = float(self.fx)
                response_data["fy"] = float(self.fy)
                response_data["cx"] = float(self.cx)
                response_data["cy"] = float(self.cy)
                
                self.get_logger().info(f"Successfully resolved plane geometry for ID: {req_id}")
            else:
                self.get_logger().warn(f"Failed plane hit check (infinite or NaN geometry) for ID: {req_id}")
        else:
            self.get_logger().warn(f"ZED SDK failed plane detection hit search for ID: {req_id}")

        res_msg = String(data=json.dumps(response_data))
        self.plane_res_pub.publish(res_msg)

    def destroy_node(self):
        self.get_logger().info("Releasing ZED camera hardware links...")
        try:
            self.zed.close()
        except Exception:
            pass
        if self.show_visualization:
            cv.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ZedDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
