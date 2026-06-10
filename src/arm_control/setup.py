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
    ],
    install_requires=['setuptools','lerobot'],
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
        ],
    },
)
