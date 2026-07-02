import os
import json
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as ScipyRot

from arm_control.utilities.fk import space_product_of_exponentials
from arm_control.utilities.se3 import screw_axis_from_w_q

class EyeToHandCalibrationNode(Node):
    def __init__(self):
        super().__init__('eye_to_hand_calibration')

        # Define Robot Kinematic Configuration (SO-101 Arm)
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

        # Declare Offset Parameters for ArUco board relative to End Effector
        self.declare_parameter('tag_offset_x', 0.0)
        self.declare_parameter('tag_offset_y', 0.0)
        self.declare_parameter('tag_offset_z', 0.0)
        self.declare_parameter('output_path', '/home/connor/robotics_projects/surgical-arm-ros2/calibration.json')

        self.tag_offset = np.array([
            self.get_parameter('tag_offset_x').value,
            self.get_parameter('tag_offset_y').value,
            self.get_parameter('tag_offset_z').value
        ])

        # Subscribers
        self.tag_sub = self.create_subscription(
            PoseStamped, 'camera/tag_pose', self.tag_callback, 10)
        self.joint_sub = self.create_subscription(
            Float32MultiArray, '/arm/current_joint_angles', self.joint_callback, 10)

        # Buffers for averaging
        self.target_samples = 20
        self.tag_poses = []
        self.joint_angles = []
        self.latest_joint_msg = None

        self.get_logger().info("Calibration Node Initialized. Please move the arm so the ArUco board is visible to the ZED camera.")

    def joint_callback(self, msg):
        self.latest_joint_msg = msg

    def tag_callback(self, msg):
        if self.latest_joint_msg is None:
            self.get_logger().warn("Tag detected but no joint feedback received yet. Skipping.", throttle_duration_sec=3.0)
            return

        # Record sample
        self.tag_poses.append(msg)
        self.joint_angles.append(list(self.latest_joint_msg.data))

        sample_count = len(self.tag_poses)
        self.get_logger().info(f"Collected sample {sample_count}/{self.target_samples}...", throttle_duration_sec=1.0)

        if sample_count >= self.target_samples:
            self.perform_calibration()

    def perform_calibration(self):
        self.get_logger().info("Sufficient samples collected. Computing calibration matrix...")

        # 1. Average Joint Angles & compute forward kinematics T_R_EE
        avg_joint_angles = np.mean(self.joint_angles, axis=0)
        theta_rad = np.radians(avg_joint_angles)
        T_R_EE = space_product_of_exponentials(self.M, self.S_list, theta_rad)

        # 2. Define board mounting offset on gripper T_EE_Tag
        T_EE_Tag = np.eye(4)
        T_EE_Tag[:3, 3] = self.tag_offset

        # T_R_Tag
        T_R_Tag = T_R_EE @ T_EE_Tag

        # 3. Average the tag pose (T_C_opt_Tag) in OpenCV optical frame
        translations = []
        rotations = []
        for pose_msg in self.tag_poses:
            p = pose_msg.pose.position
            translations.append([p.x, p.y, p.z])
            
            q = pose_msg.pose.orientation
            r = ScipyRot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            rotations.append(r)

        avg_translation = np.mean(translations, axis=0)
        # Average rotation matrices and project back to SO(3) via SVD
        avg_rotation = np.mean(rotations, axis=0)
        U, _, Vt = np.linalg.svd(avg_rotation)
        clean_rotation = U @ Vt

        T_C_opt_Tag = np.eye(4)
        T_C_opt_Tag[:3, :3] = clean_rotation
        T_C_opt_Tag[:3, 3] = avg_translation

        # 4. Transform optical frame to ZED Z-UP frame
        # ZED frame: X-right, Y-forward, Z-up
        # Optical frame: X-right, Y-down, Z-forward
        # R_opt_to_zed = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        T_zed_opt = np.array([
            [1.0,  0.0, 0.0, 0.0],
            [0.0,  0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0,  0.0, 0.0, 1.0]
        ])

        T_C_zed_Tag = T_zed_opt @ T_C_opt_Tag

        # 5. Compute Camera-to-Robot transform: T_R_C = T_R_Tag * (T_C_zed_Tag)^-1
        T_R_C = T_R_Tag @ np.linalg.inv(T_C_zed_Tag)

        # 6. Save T_R_C to calibration.json
        output_file = self.get_parameter('output_path').value
        try:
            calibration_data = {
                "T_R_C": T_R_C.tolist(),
                "avg_joint_angles": avg_joint_angles.tolist(),
                "avg_translation_opt": avg_translation.tolist()
            }
            with open(output_file, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            self.get_logger().info(f"CALIBRATION SUCCESSFUL! Saved to {output_file}")
            print("\n==========================================")
            print("Eye-to-Hand Calibration Matrix T_R_C:")
            print(T_R_C)
            print("==========================================\n")
        except Exception as e:
            self.get_logger().error(f"Failed to save calibration file: {e}")

        # Stop node processing
        self.destroy_subscription(self.tag_sub)
        self.destroy_subscription(self.joint_sub)
        self.get_logger().info("Calibration process finished. You can now press Ctrl+C to terminate this node.")

def main(args=None):
    rclpy.init(args=args)
    node = EyeToHandCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
