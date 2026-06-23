import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from std_srvs.srv import Trigger
from haplink import Haplink, DataType

class ForceSensorNode(Node):
    def __init__(self):
        super().__init__('force_sensor')
        
        # --- Parameters ---
        port = '/dev/ttyUSB0'
        baud = 115200 
        self.g = 9.80665  # m/s^2, standard gravity
        self.frequency = 100.0  # Hz, for the serial read loop

        self.scale = 77500.0  # Initial scale, will be updated when you Calibrate
        self.offset = 13997.0  # Initial offset, will be updated when you Tare
        
        self.tare_requested = False
        self.calibration_requested = False

        self.haplink = Haplink(port, baudrate=baud, timeout=0.001)

        # initalize the connection to the Haplink device
        self.get_logger().info(f"Connecting to hardware via Haplink on {port}...")
        try:
            if not self.haplink.connect():
                self.get_logger().error("Haplink connection timed out.")
                raise RuntimeError("Haplink connection failed.")            
            # Register Telemetry IDs matching the Arduino firmware
            self.haplink.register_telemetry(0, 'raw_adc', DataType.INT32)
            self.haplink.register_telemetry(1, 'arduino_time', DataType.INT32)
        except Exception as e:
            self.get_logger().error(f"Haplink failed to initialize: {str(e)}")
            raise RuntimeError("Haplink connection failed.")

        # --- ROS 2 Publishers & Services ---
        self.telemetry_pub = self.create_publisher(DiagnosticStatus, 'force_sensor/data', 10)
        
        # Universal Tare Service
        self.tare_srv = self.create_service(Trigger, 'force_sensor/tare', self.handle_tare_service)

        # calibrate service assuming the use of a 200g weight for calibration
        self.calibrate_srv = self.create_service(Trigger, 'force_sensor/calibrate', self.handle_calibrate_service)

        # Hook for dynamic command line tuning ('ros2 param set ...')
        self.add_on_set_parameters_callback(self.handle_parameter_updates)

        # --- High Speed Serial Execution loop (100Hz) ---
        self.create_timer(1/self.frequency, self.read_serial_loop)

    def read_serial_loop(self):
        try:
            self.haplink.update()
            
            raw_adc_val = self.haplink.get_telemetry('raw_adc')
            arduino_time_val = self.haplink.get_telemetry('arduino_time')
            
            if raw_adc_val is not None and arduino_time_val is not None:
                raw_adc = float(raw_adc_val)
                arduino_time = float(arduino_time_val)
                
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
                msg.name = "Force Sensor Data"
                msg.hardware_id = "hx711_pc_calibrated"
                msg.level = DiagnosticStatus.OK
                msg.message = "Streaming data from force sensor."

                msg.values = [
                    KeyValue(key="newtons", value=f"{force_newtons:.4f}"),
                    KeyValue(key="raw_adc", value=f"{raw_adc:.1f}"),
                    KeyValue(key="tared_adc", value=f"{tared_adc:.1f}"),
                    KeyValue(key="active_offset", value=f"{self.offset:.1f}"),
                    KeyValue(key="active_scale", value=f"{self.scale:.4f}"),
                    KeyValue(key="arduino_millis", value=f"{arduino_time:.1f}")
                ]
                self.telemetry_pub.publish(msg)
                
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
        if hasattr(self, 'haplink'):
            self.haplink.disconnect()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ForceSensorNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Node execution interrupted: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()