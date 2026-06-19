import sys
import pyzed.sl as sl
import cv2
from ZED_objecttracking import get_plane_at_pixel, visualize_plane

current_mouse = (0, 0)

def mouse_cb(event, x, y, flags, param):
    global current_mouse
    current_mouse = (x, y)
def main():
    zed = sl.Camera()
    
    # 1. Updated Init Parameters
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.VGA
    init.depth_mode = sl.DEPTH_MODE.NEURAL  # Replaces ULTRA
    
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED")
        sys.exit(1)

    # 2. Updated Positional Tracking
    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)
    
    # 3. Updated Spatial Mapping
    mapping_params = sl.SpatialMappingParameters()
    # If 'resolution' fails, SDK versions often use 'resolution_meter' 
    # or a specific method to set it. Try this approach:
    mapping_params.resolution_meter = 0.05 
    mapping_params.map_type = sl.SPATIAL_MAP_TYPE.MESH
    
    zed.enable_spatial_mapping(mapping_params)
    
    # ... rest of your loop ...
    cv2.namedWindow("ZED Plane Tester")
    cv2.setMouseCallback("ZED Plane Tester", mouse_cb)

    img = sl.Mat()
    runtime = sl.RuntimeParameters()

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(img, sl.VIEW.LEFT)
            frame = cv2.cvtColor(img.get_data(), cv2.COLOR_BGRA2BGR)
            
            # Detect and Visualize
            plane = get_plane_at_pixel(zed, current_mouse[0], current_mouse[1])
            frame = visualize_plane(frame, plane)
            
            cv2.imshow("ZED Plane Tester", frame)
            if cv2.waitKey(1) == ord('q'): break

    zed.close()

if __name__ == "__main__":
    main()