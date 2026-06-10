import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/mnt/c/Users/conno/OneDrive/Documents/Github/surgical-arm-ros2/install/arm_control'
