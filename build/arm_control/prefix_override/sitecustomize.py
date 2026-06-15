import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/connor/robotics_projects/surgical-arm-ros2/install/arm_control'
