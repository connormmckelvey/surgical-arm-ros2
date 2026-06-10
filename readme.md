every package contains

    package.xml file containing meta information about the package

    resource/<package_name> marker file for the package

    setup.cfg is required when a package has executables, so ros2 run can find them

    setup.py containing instructions for how to install the package

    <package_name> - a directory with the same name as your package, used by ROS 2 tools to find your package, contains __init__.py