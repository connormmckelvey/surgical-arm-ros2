import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point

# Import your custom robotics utilities
from arm_control.utilities.fk import *
from arm_control.utilities.se3 import *
from arm_control.utilities.jacobian import *
from arm_control.utilities.so3 import *
from arm_control.utilities.RR_IK import numerical_inverse_kinematics_position


class LerobotMotionPlannerNode(Node):
    def __init__(self):
        super().__init__('lerobot_motionplan')

        # 1. Define Robot Kinematic Configuration (SO-101 Arm)
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
        
        # Convert to body frame
        self.B_list = [adjoint(np.linalg.inv(self.M)) @ S for S in self.S_list]

        # Joint Limits (Converted to Radians)
        self.theta_max = np.array([105, 105, 95, 90, 90, 90]) * np.pi / 180.0
        self.theta_min = np.array([-105, -95, -90, -90, -90, -90]) * np.pi / 180.0

        # Track the latest known state of the arm (in degrees)
        self.current_joint_angles = np.array([0.0, -105.0, 95.0, -90.0, 0.0, 0.0], dtype=float)

        # 2. ROS2 Publishers and Subscribers
        self.target_joint_pub = self.create_publisher(
            Float32MultiArray,
            '/arm/target_joint_angles',
            10
        )

        self.feedback_sub = self.create_subscription(
            Float32MultiArray,
            '/arm/current_joint_angles',
            self.feedback_callback,
            10
        )

        self.cartesian_sub = self.create_subscription(
            Point,
            '/arm/target_cartesian_pose',
            self.cartesian_callback,
            10
        )

        self.get_logger().info("Motion Planner Node initialized. Listening on /arm/target_cartesian_pose")

    def feedback_callback(self, msg):
        """ Keeps internal joint state synchronized with hardware reality. """
        self.current_joint_angles = np.array(msg.data, dtype=float)

    def cartesian_callback(self, msg):
        p_des = np.array([msg.x, msg.y, msg.z], dtype=float)
        self.get_logger().info(f"Received target destination: X={msg.x:.3f}, Y={msg.y:.3f}, Z={msg.z:.3f}")
        self.command_cartesian_position(p_des)

    def compute_forward_kinematics(self):
        theta_deg = np.copy(self.current_joint_angles)
        theta_deg[5] = 0.0  # Keep gripper fixed during base calculation
        theta_rad = np.radians(theta_deg)

        T_base_to_ee = space_product_of_exponentials(self.M, self.S_list, theta_rad)
        return T_base_to_ee

    def command_cartesian_position(self, p_des):
        # 1. FIXED: Isolate gripper state from mathematical solver seed
        safe_init_angles = np.copy(self.current_joint_angles)
        gripper_memory = safe_init_angles[5]  # Preserve user gripper state
        safe_init_angles[5] = 0.0             

        theta_init_rad = np.radians(safe_init_angles)

        # Execute numerical IK solver loop
        theta_sol_rad, success = numerical_inverse_kinematics_position(
            M_ee=self.M,
            B_list=self.B_list,
            theta_init=theta_init_rad,
            p_des=p_des,
            max_iters=100,
            tol_converge=1e-6,
            tol_manipulability=1e-3,
            q_min=self.theta_min,
            q_max=self.theta_max,
            k_null=0.1,
            k_damping=0.01,
            print_iterations=False,
        )

        # 2. FIXED: Safeguard check against out-of-bounds calculations
        if not success:
            self.get_logger().error(f"IK failed for target position {p_des.tolist()}! Command dropped.")
            return

        # Convert back to degrees for the LeRobot driver
        theta_deg = np.degrees(np.asarray(theta_sol_rad, dtype=float))
        
        # Restore the physical gripper angle back onto the target packet
        theta_deg[5] = gripper_memory

        # Build and publish joint command array
        msg = Float32MultiArray()
        msg.data = theta_deg.tolist()
        self.target_joint_pub.publish(msg)
        self.get_logger().info(f"Successfully sent IK-resolved joint angles to driver node.")

    def command_joint_angles(self, angles_deg):
        msg = Float32MultiArray()
        msg.data = list(angles_deg)
        self.target_joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LerobotMotionPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()