# Surgical Arm ROS2 Modular Architecture & Visual Calibration Guide

This document provides a comprehensive system architecture, node directory, ROSbag data format reference, and mathematical specification for the visual marker calibration pipeline within the `surgical-arm-ros2` repository.

---

## 1. System Architecture Overview

The system is designed around modular ROS2 nodes that separate hardware interfaces (ZED 2i camera, Haplink serial sensor, SO-101 arm follower bus) from kinematic math solvers and pose transformation layers. This allows the system to easily switch between physical hardware control and Rviz-simulated execution.

### System Data Flow Directory

```mermaid
graph TD
    %% Input Layer
    ZED[ZED 2i Camera] -->|sl.Bodies / sl.Plane| CamNode[camera_training]
    Hap[HX711 Arduino Board] -->|Serial/Haplink| ForceNode[force_sensor]
    
    %% Processing & Transform Layer
    CamNode -->|camera/human_arm_pose <br> PoseArray| TransNode[teleop_transformer]
    CamNode -->|camera/normalized_hand_position <br> Point| BagRecord[rosbag2 Recorder]
    ForceNode -->|force_sensor/data <br> DiagnosticStatus| BagRecord
    
    %% Target Pose Generation
    TransNode -->|/arm/target_cartesian_pose <br> Point| Planner[lerobot_motionplan]
    
    %% Kinematic Core & Driver Layer
    Planner -->|/arm/target_joint_angles <br> Float32MultiArray| DriverNode[lerobot_driver / lerobot_sim]
    DriverNode -->|/arm/current_joint_angles <br> Float32MultiArray| Planner
    
    %% Visualizations
    CamNode -->|camera/visualization <br> Marker| RViz[RViz2 Debugger]
    DriverNode -->|/arm/simulated_hardware_mesh <br> Marker| RViz
```

---

## 2. Node Reference Directory

### `camera_training`
* **Source:** [camera_training.py](file:///home/connor/robotics_projects/surgical-arm-ros2/src/arm_control/arm_control/camera_training.py)
* **Description:** Interfacers with the ZED 2i camera to extract surface plane geometry and perform real-time body skeleton tracking.
* **Key Tasks:**
  * Stitches clicked surface points together using Open3D's ConvexHull to create a coordinate origin (Centroid) and normal vector.
  * Measures hand position in camera space and publishes the hand coordinates relative to the calculated plane Centroid.
  * Orchestrates `rosbag2` recording in the background using `mcap` storage formats.
* **Publishers:**
  * `camera/visualization` (`visualization_msgs/msg/Marker`): Publishes 3D geometry outlines, centroid sphere, normal arrow, and hand coordinates.
  * `camera/human_arm_pose` (`geometry_msgs/msg/PoseArray`): Joint position array containing `[Shoulder, Elbow, Wrist, Hand]`.
  * `camera/normalized_hand_position` (`geometry_msgs/msg/Point`): Real-time offset between the tracked hand and the surface plane centroid.

### `teleop_transformer`
* **Source:** [teleop_transformer.py](file:///home/connor/robotics_projects/surgical-arm-ros2/src/arm_control/arm_control/teleop_transformer.py) / [training_transformer.py](file:///home/connor/robotics_projects/surgical-arm-ros2/src/arm_control/arm_control/training_transformer.py)
* **Description:** Bridges the raw human joint telemetry to valid Cartesian coordinate targets for the robot's workspace.
* **Key Tasks:**
  * Inverts the Y-axis to map camera frame orientation to the robot coordinate frame.
  * Scales down human arm movements (default scale: `0.75`).
  * Applies hard-clipped safety bounding boxes to prevent joint collisons:
    * $X \in [0.05, 0.42]$ m
    * $Y \in [-0.25, 0.25]$ m
    * $Z \in [0.02, 0.35]$ m
* **Subscribers:**
  * `camera/human_arm_pose` (`geometry_msgs/msg/PoseArray`)
* **Publishers:**
  * `/arm/target_cartesian_pose` (`geometry_msgs/msg/Point`)
  * `/arm/target_pose_marker` (`visualization_msgs/msg/Marker`): Translucent green target sphere visualizer for RViz.

### `lerobot_motionplan`
* **Source:** [lerobot_motionplan.py](file:///home/connor/robotics_projects/surgical-arm-ros2/src/arm_control/arm_control/lerobot_motionplan.py)
* **Description:** The central kinematics solver for the 6-DoF SO-101 robot arm.
* **Key Tasks:**
  * Houses the Product of Exponentials (PoE) Forward Kinematics representation.
  * Solves Inverse Kinematics (IK) numerically using a iterative Jacobian Transpose solver.
  * Enforces joint limits (in radians) before publishing commands.
* **Subscribers:**
  * `/arm/target_cartesian_pose` (`geometry_msgs/msg/Point`)
  * `/arm/current_joint_angles` (`std_msgs/msg/Float32MultiArray`)
* **Publishers:**
  * `/arm/target_joint_angles` (`std_msgs/msg/Float32MultiArray`): Array of 6 target angles in degrees.
  * `/arm/current_cartesian_pose` (`geometry_msgs/msg/Point`): Computes live FK positions and updates RViz.

### `lerobot_driver`
* **Source:** [lerobot_driver.py](file:///home/connor/robotics_projects/surgical-arm-ros2/src/arm_control/arm_control/lerobot_driver.py)
* **Description:** Low-level hardware actuator driver using the Hugging Face `lerobot` follower bus api.
* **Key Tasks:**
  * Initializes serial links over USB (typically `/dev/ttyACM0`).
  * Interpolates joint paths with step size limits (`2.0` deg maximum change per iteration) to prevent jerking.
  * Broadcasts physical encoder positions back to ROS2.
* **Subscribers:**
  * `/arm/target_joint_angles` (`std_msgs/msg/Float32MultiArray`)
* **Publishers:**
  * `/arm/current_joint_angles` (`std_msgs/msg/Float32MultiArray`)

### `lerobot_sim`
* **Source:** [lerobot_sim.py](file:///home/connor/robotics_projects/surgical-arm-ros2/src/arm_control/arm_control/lerobot_sim.py)
* **Description:** Stand-in hardware mock driver used when physical hardware is disconnected.
* **Key Tasks:**
  * Simulates the 20Hz hardware controller loop.
  * Computes the positions of all intermediate arm links and publishes them as structural visual lines.
* **Publishers:**
  * `/arm/simulated_hardware_mesh` (`visualization_msgs/msg/Marker`): cyan line-strip representing joint-to-joint arm geometry.

### `force_sensor`
* **Source:** [force_sensor.py](file:///home/connor/robotics_projects/surgical-arm-ros2/src/arm_control/arm_control/force_sensor.py)
* **Description:** Captures high-frequency data from a load cell connected via a Haplink serial USB node.
* **Services:**
  * `force_sensor/tare` (`std_srvs/srv/Trigger`): Recalculates zero-load offset.
  * `force_sensor/calibrate` (`std_srvs/srv/Trigger`): Dynamically sets scaling factors using a standard calibration weight (default: 200g).

---

## 3. Data Recording & ROSbags

Demonstration datasets are stored inside the `training_bags` directory. The recorder uses the high-performance **MCAP** file system.

### Recorded Bag Schema
* **Format:** MCAP (`.mcap`)
* **ROS Distro:** Jazzy
* **Target Topics:**
  * `/camera/normalized_hand_position` (`geometry_msgs/msg/Point`) ~10 Hz
  * `/camera/human_arm_pose` (`geometry_msgs/msg/PoseArray`) ~10 Hz
  * `/camera/visualization` (`visualization_msgs/msg/Marker`) ~20 Hz

> [!NOTE]
> The current datasets stored in `training_bags` do not contain recorded `force_sensor/data` or robot joint angle topics. To collect full-state action trajectories, you must execute playback while recording joint encoder outputs.

---

## 4. Visual Marker Eye-to-Hand Calibration

To map hand positions from the camera-defined plane Centroid to targets that the robot's kinematics planner can execute, we define a static coordinate transform.

### Mathematical Framework

Let:
* $C$ be the Camera coordinate frame.
* $R$ be the Robot base coordinate frame.
* $EE$ be the End-Effector coordinate frame.
* $Tag$ be the physical AprilTag/ArUco frame mounted on the robot gripper.
* $Centroid$ be the calibration plane coordinate frame.

We seek the static transformation matrix:

$$T_{R}^{\text{Centroid}} = \begin{bmatrix} R_{3 \times 3} & \vec{t}_{3 \times 1} \\ \vec{0}^T & 1 \end{bmatrix}$$

Using the mounting offset of the tag on the gripper ($T_{EE}^{Tag}$) and the robot's Forward Kinematics ($T_{R}^{EE}$), the tag position in robot coordinates is:

$$T_{R}^{Tag} = T_{R}^{EE} \cdot T_{EE}^{Tag}$$

Simultaneously, the camera views the tag, giving its position relative to the camera ($T_{C}^{Tag}$). The static camera-to-robot transform ($T_{R}^{C}$) is solved via:

$$T_{R}^{C} = T_{R}^{Tag} \cdot \left(T_{C}^{Tag}\right)^{-1}$$

With $T_{R}^{C}$ established, we map the centroid to the robot base:

$$T_{R}^{\text{Centroid}} = T_{R}^{C} \cdot T_{C}^{\text{Centroid}}$$

For any normalized coordinate $\vec{p}_{\text{Centroid}}$ streamed from your demonstration bags, the physical target point $\vec{p}_{R}$ in the robot workspace is:

$$\vec{p}_{R} = T_{R}^{\text{Centroid}} \cdot \vec{p}_{\text{Centroid}}$$

---

## 5. Execution Workflow

### Local Compilation

```bash
# Navigate to workspace root
cd /home/connor/robotics_projects/surgical-arm-ros2

# Source python environment and build packages
source venv/bin/activate
colcon build

# Sourcing workspace underlay
source install/setup.bash
```

### Running Simulated Teleoperation & Debugging
Run the following commands in separate terminals to spin up the simulation workspace:

1. **Launch Simulated Driver:**
   ```bash
   ros2 run arm_control lerobot_sim
   ```
2. **Launch Motion Planner:**
   ```bash
   ros2 run arm_control lerobot_motionplan
   ```
3. **Launch Teleop Coordinate Transformer:**
   ```bash
   ros2 run arm_control teleop_transformer
   ```
4. **Launch RViz Visualizer:**
   ```bash
   rviz2
   ```
5. **Play Demonstration Rosbag:**
   ```bash
   ros2 bag play training_bags/training_episode_20260622_174229/
   ```
