from setuptools import find_packages, setup

package_name = 'arm_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/teleop.launch.py',
            'launch/playback.launch.py',
            'launch/training.launch.py'
        ]),
    ],
    install_requires=['setuptools','lerobot', 'numpy', 'opencv-python', 'pyzed'],
    zip_safe=True,
    maintainer='connor',
    maintainer_email='connor@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lerobot_driver = arm_control.lerobot_driver:main',
            'lerobot_motionplan = arm_control.lerobot_motionplan:main',
            'camera_training = arm_control.camera_training:main',
            'camera_playback = arm_control.camera_playback:main',
            'teleop_transformer = arm_control.teleop_transformer:main',
            'lerobot_sim = arm_control.lerobot_sim:main',
            'force_sensor = arm_control.force_sensor:main',
            'zed_driver = arm_control.zed_driver:main',
            'eye_to_hand_calibration = arm_control.eye_to_hand_calibration:main',
            'playback_transformer = arm_control.playback_transformer:main',
        ],
    },
)
