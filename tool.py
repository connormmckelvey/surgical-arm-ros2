import sys
import termios
import tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

# Text UI formatting
msg = """
---------------------------------------------
LeRobot Cartesian Keyboard Teleop Controller
---------------------------------------------
Moving instructions:
        W                      Q (Up)
   A    S    D                 E (Down)

W/S : Increase/Decrease X (Forward/Backward)
A/D : Increase/Decrease Y (Left/Right)
Q/E : Increase/Decrease Z (Up/Down)

SPACEBAR : Jump to Home Position
CTRL+C   : Quit safely
---------------------------------------------
"""

class LeRobotTeleopNode(Node):
    def __init__(self):
        super().__init__('lerobot_teleop')

        # Default Home Position (matching your planner's M matrix)
        self.home_x = 0.391
        self.home_y = 0.0
        self.home_z = 0.243

        # Track target positions
        self.target_x = self.home_x
        self.target_y = self.home_y
        self.target_z = self.home_z

        # Step size per keypress (0.01 meters = 1 cm)
        self.step_size = 0.01 

        # Publishers and Subscribers
        self.target_pub = self.create_publisher(Point, '/arm/target_cartesian_pose', 10)
        self.current_sub = self.create_subscription(Point, '/arm/current_cartesian_pose', self.current_pose_callback, 10)

        self.get_logger().info("Teleop Node Initialized. Awaiting current arm feedback...")

    def current_pose_callback(self, msg):
        """
        Keeps our teleop targets anchored to where the arm actually is 
        so it doesn't 'jump' on the first keypress.
        """
        # Only snap to current position if we aren't actively processing commands
        # to avoid feedback loops while moving.
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_z = msg.z

    def sync_to_actual(self):
        """ Syncs target internal values to actual current position if available """
        if hasattr(self, 'current_x'):
            self.target_x = self.current_x
            self.target_y = self.current_y
            self.target_z = self.current_z

    def update_and_publish(self, char):
        # Sync with actual hardware positions before calculating offset
        self.sync_to_actual()

        if char == 'w':
            self.target_x += self.step_size
        elif char == 's':
            self.target_x -= self.step_size
        elif char == 'a':
            self.target_y += self.step_size
        elif char == 'd':
            self.target_y -= self.step_size
        elif char == 'q':
            self.target_z += self.step_size
        elif char == 'e':
            self.target_z -= self.step_size
        elif char == ' ':
            self.target_x = self.home_x
            self.target_y = self.home_y
            self.target_z = self.home_z
            self.get_logger().info("Commanding Home Position...")
        else:
            return # Unmapped key

        # Create and publish message
        point_msg = Point()
        point_msg.x = float(self.target_x)
        point_msg.y = float(self.target_y)
        point_msg.z = float(self.target_z)
        
        self.target_pub.publish(point_msg)
        print(f"Target Sent -> X: {self.target_x:.3f} | Y: {self.target_y:.3f} | Z: {self.target_z:.3f}", end='\r')


def get_key(settings):
    """ Reads raw characters from the terminal stdin without blocking or needing Enter """
    tty.setraw(sys.stdin.fileno())
    # select allows us to check if input is waiting
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):
    # Save terminal settings so we can restore them when quitting
    old_settings = termios.tcgetattr(sys.stdin)

    rclpy.init(args=args)
    node = LeRobotTeleopNode()

    print(msg)

    try:
        while rclpy.ok():
            # Spin ROS background callbacks to keep current pose updated
            rclpy.spin_once(node, timeout_sec=0.01)
            
            # Read keyboard input
            key = get_key(old_settings)
            
            # Catch Ctrl+C escape character
            if key == '\x03':
                break
                
            node.update_and_publish(key.lower())

    except Exception as e:
        print(f"\nError in teleop loop: {e}")
        
    finally:
        # CRITICAL: Always restore terminal settings or your WSL terminal will break!
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()
        print("\nTeleop node shutdown cleanly.")

if __name__ == '__main__':
    main()