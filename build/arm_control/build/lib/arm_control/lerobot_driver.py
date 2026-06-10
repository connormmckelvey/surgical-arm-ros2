import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


class LeRobotDriverNode(Node):
    def __init__(self):
        super().__init__('lerobot_driver')        
        
        # Order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        self.home_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.rest_position = np.array([0.0, -105.0, 95.0, -90.0, 0.0, 0.0], dtype=float)
        self.max_step_deg = 2.0
        self.control_loop_period = 0.05  # 20Hz background loop

        # Physical connection to LeRobot
        config = SO101FollowerConfig(port="/dev/ttyACM0", id="dbot")
        self.robot = SO101Follower(config)
        self.robot.connect(calibrate=False)
        self.get_logger().info("Hardware connection established over LeRobot follower bus.")

        # Dictionary keys required by the underlying LeRobot low-level driver
        self.lerobot_keys = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
        ]

        self.current_positions = np.copy(self.rest_position)
        self.target_positions = np.copy(self.current_positions)

        # ROS2 Subscriber and Publisher
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

        # Asynchronous Timer (Runs the hardware loop at 20Hz without blocking ROS)
        self.control_timer = self.create_timer(self.control_loop_period, self.control_loop_callback)

    """
        when something is published, make sure its complete with six joints, then set target_positions member to these new values.
    """
    def joint_angle_callback(self, msg):
        input_angles = np.array(msg.data, dtype=float)

        if len(input_angles) != 6:
            self.get_logger().error(f"Expected 6 joint angles, received: {len(input_angles)}")
            return

        # Update the target state asynchronously
        self.target_positions = input_angles

    """
        asynchronously updates the hardware state at a fixed rate.
    """
    def control_loop_callback(self):
        diff = self.target_positions - self.current_positions
        max_diff = np.max(np.abs(diff))
        if max_diff > 1e-9:
            # Scale down the movement step if it exceeds our maximum safe velocity threshold
            step_scale = min(1.0, self.max_step_deg / max_diff)
            self.current_positions += diff * step_scale

            # Command the physical hardware using the required dictionary mapping
            action = {
                name: float(self.current_positions[idx])
                for idx, name in enumerate(self.lerobot_keys)
            }
            self.robot.send_action(action)

        #publish current state for visualization and feedback
        
        state_msg = Float32MultiArray()
        state_msg.data = self.current_positions.tolist()
        self.current_state_pub.publish(state_msg)

    def destroy_node(self):
        self.get_logger().info("Safely severing hardware bus connections...")
        self.robot.disconnect()
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