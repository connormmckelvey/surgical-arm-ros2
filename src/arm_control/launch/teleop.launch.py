import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    sim_arg = DeclareLaunchArgument(
        'sim',
        default_value='false',
        description='Set to true to run in simulation mode using lerobot_sim'
    )

    sim_value = LaunchConfiguration('sim')

    # 1. Driver (Simulation or Real Hardware)
    driver_sim_node = Node(
        package='arm_control',
        executable='lerobot_sim',
        name='lerobot_driver', # Keeps same name for namespace parity
        output='screen',
        condition=IfCondition(sim_value)
    )

    driver_real_node = Node(
        package='arm_control',
        executable='lerobot_driver',
        name='lerobot_driver',
        output='screen',
        condition=UnlessCondition(sim_value)
    )

    # 2. Camera Driver
    camera_node = Node(
        package='arm_control',
        executable='zed_driver',
        name='zed_driver',
        output='screen'
    )

    # 3. Motion Planner
    planner_node = Node(
        package='arm_control',
        executable='lerobot_motionplan',
        name='lerobot_motionplan',
        output='screen'
    )

    # 4. Teleop Transformer
    transformer_node = Node(
        package='arm_control',
        executable='teleop_transformer',
        name='teleop_transformer',
        output='screen'
    )

    return LaunchDescription([
        sim_arg,
        driver_sim_node,
        driver_real_node,
        camera_node,
        planner_node,
        transformer_node
    ])
