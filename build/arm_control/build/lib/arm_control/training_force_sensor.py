import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from std_srvs.srv import Trigger
import serial

class TrainingForceSensorNode(Node):
    def __init__(self):
        super().__init__('training_force_sensor')
        
        # --- Declare ROS 2 Parameters ---
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('gravity_constant', 9.80665)
        self.declare_parameter('frequency', 100.0)  # Hz, for the serial read loop
        
        # Runtime calibration parameters managed inside ROS
        self.declare_parameter('scale', 77500.0)       # Set this after your calibration test
        self.declare_parameter('offset', 13997.0)      # Will be updated automatically when you Tare
        
        # Fetch initial configurations
        port = self.get_parameter('serial_port').get_parameter_value().string_value
        baud = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.g = self.get_parameter('gravity_constant').get_parameter_value().double_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value

        self.scale = self.get_parameter('scale').get_parameter_value().double_value
        self.offset = self.get_parameter('offset').get_parameter_value().double_value
        
        self.tare_requested = False
        self.calibration_requested = False

        # --- Initialize PySerial Pipeline ---
        self.get_logger().info(f"Opening hardware serial channel on {port}...")
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=1.0)
            self.ser.reset_input_buffer() 
        except Exception as e:
            self.get_logger().error(f"Serial port failed to initialize: {str(e)}")
            raise RuntimeError("Serial connection failed.")

        # --- ROS 2 Publishers & Services ---
        self.telemetry_pub = self.create_publisher(DiagnosticStatus, 'training_sensor/data', 10)
        
        # Universal Tare Service
        self.tare_srv = self.create_service(Trigger, 'training_sensor/tare', self.handle_tare_service)

        # calibrate service assuming the use of a 200g weight for calibration
        self.calibrate_srv = self.create_service(Trigger, 'training_sensor/calibrate', self.handle_calibrate_service)

        # Hook for dynamic command line tuning ('ros2 param set ...')
        self.add_on_set_parameters_callback(self.handle_parameter_updates)

        # --- High Speed Serial Execution loop (100Hz) ---
        self.create_timer(1/self.frequency, self.read_serial_loop)

    def read_serial_loop(self):
        if self.ser.in_waiting > 0:
            try:
                raw_line = self.ser.readline()
                decoded_line = raw_line.decode('utf-8').strip()
                data_tokens = decoded_line.split(',')
                
                if len(data_tokens) == 2:
                    raw_adc = float(data_tokens[0])
                    arduino_time = float(data_tokens[1])
                    
                    # --- Check if user triggered a Tare Event ---
                    if self.tare_requested:
                        self.offset = raw_adc
                        self.tare_requested = False
                        self.get_logger().info(f"Tare complete! New zero offset calculated: {self.offset}")
                    
                    if self.calibration_requested:
                        # Assuming a known weight of 200g for calibration
                        known_mass_kg = 0.2
                        self.scale = (raw_adc - self.offset) / (known_mass_kg * self.g)
                        self.calibration_requested = False
                        self.get_logger().info(f"Calibration complete! New scale factor calculated: {self.scale}")

                    # --- Execute The Calibration Math Pipeline ---
                    tared_adc = raw_adc - self.offset
                    calculated_mass = tared_adc / self.scale
                    force_newtons = calculated_mass * self.g
                    
                    # Live Diagnostic Terminal Feedback
                    print(f"Raw: {raw_adc:.0f} | Tared: {tared_adc:.0f} | Newtons: {force_newtons:.4f} N | Time: {arduino_time:.0f}ms")
                    
                    # --- Construct and Publish Dictionary Packet ---
                    msg = DiagnosticStatus()
                    msg.name = "Training Force Sensor Data"
                    msg.hardware_id = "hx711_pc_calibrated"
                    msg.level = DiagnosticStatus.OK
                    msg.message = "Streaming data from training sensor."
    
                    msg.values = [
                        KeyValue(key="newtons", value=f"{force_newtons:.4f}"),
                        KeyValue(key="raw_adc", value=f"{raw_adc:.1f}"),
                        KeyValue(key="tared_adc", value=f"{tared_adc:.1f}"),
                        KeyValue(key="active_offset", value=f"{self.offset:.1f}"),
                        KeyValue(key="active_scale", value=f"{self.scale:.4f}"),
                        KeyValue(key="arduino_millis", value=f"{arduino_time:.1f}")
                    ]
                    self.telemetry_pub.publish(msg)
                    
            except (ValueError, IndexError):
                pass
            except Exception as e:
                self.get_logger().warn(f"error: {str(e)}")

    def handle_parameter_updates(self, params):
        result = SetParametersResult(successful=True)
        for param in params:
            if param.name == 'scale':
                self.scale = float(param.value)
                self.get_logger().info(f"Scale factor changed to: {self.scale}")
            elif param.name == 'offset':
                self.offset = float(param.value)
                self.get_logger().info(f"Offset changed to: {self.offset}")
        return result

    def handle_tare_service(self, request, response):
        self.get_logger().info("cmd received")
        self.tare_requested = True
        response.success = True
        response.message = "tare command received."
        return response
    
    def handle_calibrate_service(self, request, response):
        self.get_logger().info("Calibration command received place 200g")
        self.calibration_requested = True
        response.success = True
        response.message = "Calibration command acknowledged"
        return response

    def destroy_node(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = TrainingForceSensorNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Node execution interrupted: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()