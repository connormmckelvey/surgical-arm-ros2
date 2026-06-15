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
from arm_control.utilities.jacobian_transpose import jacobian_transpose_position

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

        # Target Cartesian Pose Subscriber
        self.cartesian_sub = self.create_subscription(
            Point,
            '/arm/target_cartesian_pose',
            self.cartesian_callback,
            10
        )

        # FIXED: Corrected standard ROS2 Publisher configuration (No callback parameter)
        self.current_cartesian_pub = self.create_publisher(
            Point,
            '/arm/current_cartesian_pose',
            10
        )

        self.get_logger().info("Motion Planner Node initialized. Listening on /arm/target_cartesian_pose")

    def feedback_callback(self, msg):
        """ Keeps internal joint state synchronized and streams real-time Cartesian feedback. """
        # update internal state from topic
        self.current_joint_angles = np.array(msg.data, dtype=float)

        # Automatically recalculate where the end-effector physically is right now
        try:
            T_base_to_ee = self.compute_forward_kinematics()
            
            # Create and publish the geometry point message
            cartesian_msg = Point()
            cartesian_msg.x = float(T_base_to_ee[0, 3])
            cartesian_msg.y = float(T_base_to_ee[1, 3])
            cartesian_msg.z = float(T_base_to_ee[2, 3])
            
            self.current_cartesian_pub.publish(cartesian_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to compute or publish forward kinematics: {e}")

    def cartesian_callback(self, msg):
        p_des = np.array([msg.x, msg.y, msg.z], dtype=float)
        self.get_logger().info(f"Received target destination: X={msg.x:.3f}, Y={msg.y:.3f}, Z={msg.z:.3f}")
        self.command_cartesian_position(p_des)

    def compute_forward_kinematics(self):
        #joint angles in degrees, convert to radians for FK calculation
        theta_deg = np.copy(self.current_joint_angles)
        theta_rad = np.radians(theta_deg)

        T_base_to_ee = space_product_of_exponentials(self.M, self.S_list, theta_rad)
        return T_base_to_ee

    def command_cartesian_position(self, p_des):
        init_angles = np.copy(self.current_joint_angles)
        #gripper_memory = safe_init_angles[5]
        #safe_init_angles[5] = 0.0             

        theta_init_rad = np.radians(init_angles)

        #compute
        theta_sol_rad, theta_sol_rad_hist = jacobian_transpose_position(
                    M_ee=self.M,
                    B_list=self.B_list,
                    theta_init=theta_init_rad,
                    p_des=p_des,
                    max_iters=100,
                    tol_converge=1e-6,
                    q_min=self.theta_min,
                    q_max=self.theta_max
        )

        # Verify if the solution is structurally valid (not NaN or infinite)
        if theta_sol_rad is None or np.isnan(theta_sol_rad).any() or np.isinf(theta_sol_rad).any():
            self.get_logger().error(f"IK engine failed to reach target coordinate: X={p_des[0]:.3f}, Y={p_des[1]:.3f}, Z={p_des[2]:.3f}! Command dropped.")
            return

        # Convert back to degrees for the LeRobot driver
        theta_deg = np.degrees(np.asarray(theta_sol_rad, dtype=float))
        
        # Restore the physical gripper angle back onto the target packet
        #theta_deg[5] = gripper_memory
        
        # Build and publish joint command array
        msg = Float32MultiArray()
        msg.data = theta_deg.tolist()
        self.target_joint_pub.publish(msg)
        self.get_logger().info(f"Successfully sent IK-resolved joint angles to driver node.")

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