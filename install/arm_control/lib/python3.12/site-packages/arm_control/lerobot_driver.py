import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


class LeRobotDriverNode(Node):
    def __init__(self):
        super().__init__('lerobot_driver')        
        
        # Motion parameters
        self.max_step_deg = 2.0
        self.control_loop_period = 0.05  # 20Hz loop
        self.gripper_offset = 50.0  # Gripper offset to ensure it starts in a neutral position

        # Order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        self.lerobot_keys = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
        ]
        
        # 1. Establish Physical Connection
        config = SO101FollowerConfig(port="/dev/ttyACM0", id="dbot")
        self.robot = SO101Follower(config)
        self.robot.connect(calibrate=True)
        self.get_logger().info("Hardware connection established with LeRobot")

        # 2. Zero Out All Joints to Home Position
        self.get_logger().info("Initializing: Commanding all joints to 0.0 Position...")
        startup_home_action = {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0, "elbow_flex.pos": 0.0,
                               "wrist_flex.pos": 0.0, "wrist_roll.pos": 0.0, "gripper.pos": 50.0}
        self.robot.send_action(startup_home_action)
        
        # Now that the physical hardware has been explicitly sent to 0, 
        self.current_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.target_positions = np.copy(self.current_positions)

        # 4. ROS2 Subscribers and Publishers
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/arm/target_joint_angles',
            self.joint_angle_callback,
            10
        )

        self.current_state_pub = self.create_publisher(
            Float32MultiArray, 
            '/arm/current_joint_angles', 
            10
        )

        # 5. Start Runtime Control Loop
        self.control_timer = self.create_timer(self.control_loop_period, self.control_loop_callback)
        self.get_logger().info("Driver successfully initialized and zeroed out.")

    def joint_angle_callback(self, msg):
        input_angles = np.array(msg.data, dtype=float)
        if len(input_angles) != 6:
            self.get_logger().error(f"Expected 6 joint angles, received: {len(input_angles)}")
            return
        self.target_positions = input_angles

    def control_loop_callback(self):
        diff = self.target_positions - self.current_positions
        max_diff = np.max(np.abs(diff))
        
        if max_diff > 1e-5:
            step_scale = min(1.0, self.max_step_deg / max_diff)
            self.current_positions += diff * step_scale

            action = {
                name: float(self.current_positions[idx])
                for idx, name in enumerate(self.lerobot_keys)
            }
            action["gripper.pos"] += self.gripper_offset  # Apply gripper offset
            self.robot.send_action(action)

        # Continually publish state back to your planner node
        state_msg = Float32MultiArray()
        state_msg.data = self.current_positions.tolist()
        self.current_state_pub.publish(state_msg)

    def destroy_node(self):
        self.get_logger().info("Safely severing hardware bus connections...")
        try:
            self.robot.disconnect()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()