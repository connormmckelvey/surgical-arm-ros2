#!/usr/bin/env python3

import os
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import pyzed.sl as sl
import cv2 as cv
import numpy as np

# ==============================================================================
# CAMERA DRIVER NODE: published x,y,z of wrist and shoulder keypoints as a PoseArray message on /camera/human_arm_pose from ZED 2i
# ==============================================================================

# REMOVED/COMMENTED OUT: This line was forcing the GUI to remain hidden!
# os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Importing utility functions from your workspace layout
from arm_control.utilities.ZED_bodytracking import (
    setup_body_tracking,
    get_single_body,
    get_arm_points,
    draw_arm_points_and_lines,
)

class ZedArmSensorNode(Node):
    def __init__(self):
        super().__init__('zed_arm_sensor')
        
        # Configure tracking target profile ("right" or "left")
        self.arm_to_track = "right"  
        
        # Poses list mapping layout: [0] = Shoulder position, [1] = Wrist position
        self.arm_pose_pub = self.create_publisher(PoseArray, 'camera/human_arm_pose', 10)
        
        # ZED camera initialization parameters
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        
        # Lowered camera fps bc my computer was exploding
        init_params.camera_resolution = sl.RESOLUTION.VGA  
        init_params.camera_fps = 15
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

        self.get_logger().info("Attempting link establishment with ZED 2i hardware channels...")
        
        # Establish hardware hook
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(
                "CRITICAL: Failed to open ZED camera interface!"
            )
            raise RuntimeError("ZED initialization failed.")
            
        # Spin up internal neural spatial network parameters
        self.body_runtime = setup_body_tracking(self.zed)
        self.image = sl.Mat()
        self.bodies = sl.Bodies()
        self.runtime = sl.RuntimeParameters()
        
        # Set up a timer to process frames at approximately the camera's FPS rate
        self.timer = self.create_timer(0.066, self.process_frame_callback)
        self.get_logger().info("Camera driver node initialized, successfully connected to ZED camera.")

    def process_frame_callback(self):
        try:
            grab_status = self.zed.grab(self.runtime)
            if grab_status != sl.ERROR_CODE.SUCCESS:
                return
        except Exception as e:
            self.get_logger().warn(f"USB interface dropped packets: {str(e)}. Attempting recovery sweep...")
            return

        # Fetch visual and point-cloud array data matrix structures
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        frame = self.image.get_data()
        if frame.shape[2] == 4:
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

        # Retrieve AI keypoint skeletal positions
        self.zed.retrieve_bodies(self.bodies, self.body_runtime)
        body = get_single_body(self.bodies, mode="closest")

        if body is not None:
            latest_arm_data = get_arm_points(body, arm=self.arm_to_track)
            
            if latest_arm_data is not None:
                # Render tracking canvas modifications internally (draws the skeleton/joints)
                frame = draw_arm_points_and_lines(frame, latest_arm_data)
                
                sh_xyz = latest_arm_data["shoulder_3d"]
                wr_xyz = latest_arm_data["wrist_3d"]
                
                # --- Construct and Publish ROS2 Message Packet ---
                msg = PoseArray()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "zed_camera_frame"
                
                # Element [0]: Shoulder Pose Transform Map
                shoulder_pose = Pose()
                shoulder_pose.position.x = float(sh_xyz[0])
                shoulder_pose.position.y = float(sh_xyz[1])
                shoulder_pose.position.z = float(sh_xyz[2])
                msg.poses.append(shoulder_pose)
                
                # Element [1]: Wrist Pose Transform Map
                wrist_pose = Pose()
                wrist_pose.position.x = float(wr_xyz[0])
                wrist_pose.position.y = float(wr_xyz[1])
                wrist_pose.position.z = float(wr_xyz[2])
                msg.poses.append(wrist_pose)
                
                self.arm_pose_pub.publish(msg)

        # NEW: Pop open a live visual window showing the feed + overlays
        cv.imshow("ZED 2i Tracking Feed", frame)

        # OpenCV requires an explicit frame clock release statement to purge 
        # its internal cache memory matrices, preventing a runtime memory leakage crash.
        cv.waitKey(1)

    def destroy_node(self):
        self.get_logger().info("Safely severing camera interface connections.")
        try:
            self.zed.close()
            cv.destroyAllWindows() # NEW: Closes the desktop window cleanly on shutdown
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