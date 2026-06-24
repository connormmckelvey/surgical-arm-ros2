# Surgical Cleaning Robot ROS2 Migration
Written by Connor McKelvey in collaberation with Daniel Agbara, HERO Lab Texas Robotics

### General Project Structure

This project was built in python using the ROS2 framework, the main goal of the migration was to modularize the code and seperate the core math from hardware to prepare for a hardware migration to a more advanced arm and future humanoids. The project is built in one package currently called `arm_control` this holds six nodes with different features. The package relies on a few external packages listed in the `setup.py` including pyzed, lerobot, open3d and there dependencies. 

### Usage

On Ubuntu 24: 

    """
        first setup a venv and install required packages, 
        then source your ros2 build underlayer
    """ 
    colcon build //to build changes
    source install/setup.bash //to source changes
    ros2 run arm_control <node_name> //run a given node
    ros2 pkg executables //get a list of built node names
    ros2 topic echo <topic> //listen to a topic to see what is sent
    ros2 run rviz2 rivz2 //run visualizer
    ros2 bag play <dir_name_of_rosbag>
    ros2 service call /force_sensor/tare std_srvs/srv/Trigger //tare the force_sensor


Note: for hardware if you are running on a windows machine in wsl you will need to attach various hardware ports

### Nodes

- `lerobot_driver` is a hardware driver for the soarm101 which connects to the hardware port and can be interfaced through the below topics
    - `arm/target_joint_angles` subscribes, listening for `Float32MultiArray` messages containing the joint angles for the `self.lerobot_keys`
    - `arm/current_joint_angles` publishes a `Float32MultiArray` of the current joint angles.
    <br>
- `lerobot_sim` is a node designed to run in place of the `lerobot_dirver` it subscribes and publishes to the same topics but instead of connected to real hardware it updates the `arm/simulated_hardware_mesh` a topic of `Marker` message type to be visualized using rviz2
    <br>
- `lerobot_motionplan` uses the math modules in the utilities folder to perform IK and translate 3d points to joint angles. 
    - `arm/target_joint_angles/` and `arm/current_joint_angles` publishes to the first and reads from the latter to allow it to keep the current 3d position
    - `arm/target_cartesian_pose` subscribes to convert this point to joint angles, if it fails it sends a log message and ignores. It expects a message of type `Point`
    - `arm/current_cartesian_pose` publishes end effector position xyz as a `Point`
    <br>
- `teleop_transformer` takes a human arm pose and converts it in real time to a reachable point for the robot arm to travel to. This node is seperated from hardware and you will simply have to update the current 3d cube bounding box based of the arms workspace. This takes the point scales it down by a factor of 0.75 and then adds an offset to make a flat human arm be raised off the working space of the robot. it also applies a filter to prevent noise and small changes from being replicated. This transformer also flips the Y-axis as the cordinate frame of the camera (in right hand Z) and the robot frame have the Z inverted. 
    - `camera/human_arm_pose` subscribes to this as input, expects a `PoseArray` message type. Currently DOES NOT have support for hand tracking and treats the wrist as the end effector. NOT COMPATIBLE WITH CURRENT CAMERA DRIVER
    - `target_cartesian_pose` outputs point
    - `arm/target_pose_marker` a Marker for rviz2 to visualize transformed target position sometimes will be inverted do to default rviz2 coordinate frame even though its working correctly on the robot. 
    <br>
- `force_sensor` opens a hardware port the arduino which reads data from our load cell. Stores both the tare offset and scaling factor as params and performs the calibration in the node. IT DOES NOT USE THE BUILT IN FEATURES OF THE HARDWARE LIBARY. It is currently calibrated using 200g but needs to be tared on start up. 
    - `force_sensor/tare` Trigger service that tares the node
    - `force_sensor/calibrate` assumes a 200g mass was placed on the load cell and calibrates the Newton output
    - `force_sensor/data` is a `DiagnosticData` type (which should probably change) that publishes the current `newtons`,`raw_adc`,`tared_adc`,`active_offset`,`active_scale`,`arduino_millis`  
    <br>
- `camera_calibration` the most complex and least modular node of the group, it is designed to run in parallel to the `training_force_sensor` during data collection. This node lets you select planes, stitches them together using ConvexHull to create a working surface with a normal vector. It then calculates the centroid of that area and treating it as the origin. It publishes visualizations of this and a normalized hand coordinate. Finally it uses `rosbag2` to store all of this data in one place to be used later. When the node is run a window will pop up, click on planes to highlight them, and then press 's' to generate the surface and centroid this will also start the rosbag recording. 'esc' stops the recording and clears the surface. 
    - `camera/human_arm_pose` publishes a `PoseArray` including the shoulder, elbow, wrist, hand positions. 
    - `camera/normalized_hand_position` publishes a `Point` holding the hand position normalized to the centroid origin
    - `camera/visualization` publishes `Marker` messages to be viewed in rviz2 this includes a blue dot representing the centroid, a pink dot representing the hand, a blue surface representing the generated plane (often buggy), an orange arrow representing the normal vector of the surface, a green line representing the outline of the plane/surface, and a text marker showing the normalized cordinates.
