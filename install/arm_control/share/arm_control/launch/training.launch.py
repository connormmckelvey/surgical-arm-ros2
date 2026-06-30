from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. ZED Camera Driver
    camera_node = Node(
        package='arm_control',
        executable='zed_driver',
        name='zed_driver',
        output='screen'
    )

    # 2. Force Sensor Driver
    force_sensor_node = Node(
        package='arm_control',
        executable='force_sensor',
        name='force_sensor'
    )

    # 3. Training UI
    camera_training_node = Node(
        package='arm_control',
        executable='camera_training',
        name='camera_training',
        output='screen'
    )

    return LaunchDescription([
        camera_node,
        force_sensor_node,
        camera_training_node
    ])
