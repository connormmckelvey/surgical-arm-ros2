import datetime
import subprocess
import signal

def start_rosbag_recording(topics_to_record, rosbag_folder): 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = f"episode_{timestamp}"
    
    
    cmd = ["ros2", "bag", "record", "-s", "mcap", "-o", rosbag_folder + "/" + bag_name] + topics_to_record

    print(f"rosbag started: {bag_name}")
    bag_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return bag_process

def stop_rosbag_recording(bag_process):
    if bag_process is not None:
        bag_process.send_signal(signal.SIGINT)  
        bag_process.wait()                      
        bag_process = None
        print("Rosbag folder successfully generated and closed.")
        return True
    return False
