# Latest version
#pip install git+https://github.com/connormmckelvey/Haplink.git#subdirectory=python

from haplink import Haplink, DataType

class ForceSensor():
    def __init__(self):
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
        try:
            if not self.haplink.connect():
                raise RuntimeError("Haplink connection failed.")            
            # Register Telemetry IDs matching the Arduino firmware
            self.haplink.register_telemetry(0, 'raw_adc', DataType.INT32)
            self.haplink.register_telemetry(1, 'arduino_time', DataType.INT32)
        except Exception as e:
            raise RuntimeError("Haplink connection failed.")


    def read_serial_loop(self):
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
                print(f"Tare complete! New zero offset calculated: {self.offset}")
            
            if self.calibration_requested:
                # Assuming a known weight of 200g for calibration
                known_mass_kg = 0.2
                self.scale = (raw_adc - self.offset) / (known_mass_kg * self.g)
                self.calibration_requested = False
                print(f"Calibration complete! New scale factor calculated: {self.scale}")

            # --- Execute The Calibration Math Pipeline ---
            tared_adc = raw_adc - self.offset
            calculated_mass = tared_adc / self.scale
            force_newtons = calculated_mass * self.g
            
            # Live Diagnostic Terminal Feedback
            print(f"Raw: {raw_adc:.0f} | Tared: {tared_adc:.0f} | Newtons: {force_newtons:.4f} N | Time: {arduino_time:.0f}ms")
            

    def tare(self):
        self.tare_requested = True

    def handle_calibrate_service(self):
        self.calibration_requested = True

    def __del__(self):
        self.haplink.disconnect()

if __name__ == '__main__':
    force_sensor = ForceSensor()
    try:
        while True:
            force_sensor.read_serial_loop()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        del force_sensor