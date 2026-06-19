#!/usr/bin/env python3
import sys
import pyzed.sl as sl
import cv2 as cv
import numpy as np

# Global variable to store click coordinates
click_coord = None

def mouse_callback(event, x, y, flags, param):
    global click_coord
    if event == cv.EVENT_LBUTTONDOWN:
        click_coord = (x, y)
        print(f"Clicked at pixel: {x}, {y}")

def main():
    global click_coord
    zed = sl.Camera()
    init = sl.InitParameters()
    
    init.camera_resolution = sl.RESOLUTION.VGA  
    init.camera_fps = 15
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("CRITICAL: Failed to open ZED camera interface!")
        sys.exit(1)

    zed.enable_positional_tracking(sl.PositionalTrackingParameters())
    
    # Extract Camera Intrinsics for 3D -> 2D Projection
    cam_info = zed.get_camera_information()
    cam_params = cam_info.camera_configuration.calibration_parameters.left_cam
    fx, fy = cam_params.fx, cam_params.fy
    cx, cy = cam_params.cx, cam_params.cy

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    plane = sl.Plane() 
    
    cv.namedWindow("Click on a surface")
    cv.setMouseCallback("Click on a surface", mouse_callback)

    print("Successfully connected. Click on any surface in the feed to query and visualize the plane.")

    # We will store the 2D polygon here to keep it drawn on screen
    active_polygon = None

    try:
        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()
                
                if frame.shape[2] == 4:
                    frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
                
                # --- PROCESS CLICK & CALCULATE BOUNDS ---
                if click_coord is not None:
                    err = zed.find_plane_at_hit(click_coord, plane)
                    
                    if err == sl.ERROR_CODE.SUCCESS:
                        print("\n--- NEW PLANE DETECTED ---")
                        bounds = plane.get_bounds()
                        
                        # Project 3D bounds to 2D pixels
                        pts_2d = []
                        for pt in bounds:
                            # pt is [X, Y, Z] in RIGHT_HANDED_Z_UP (Camera Frame)
                            # X = Right, Y = Forward (Depth), Z = Up
                            X, Y_depth, Z = pt[0], pt[1], pt[2]
                            
                            # Ensure the point is actually in front of the camera
                            if Y_depth > 0.01:
                                # Map to OpenCV image axes (X=Right, Y=Down)
                                # Since ZED's Z is Up, we flip it to -Z for OpenCV's Down
                                u = int((X / Y_depth) * fx + cx)
                                v = int((-Z / Y_depth) * fy + cy)
                                pts_2d.append([u, v])
                        
                        # Save the polygon if valid
                        if len(pts_2d) > 2:
                            active_polygon = np.array(pts_2d, dtype=np.int32)
                        else:
                            active_polygon = None
                            
                    else:
                        print("Could not detect a plane at this pixel (try a flat surface).")
                        active_polygon = None
                    
                    click_coord = None # Reset click

                # --- DRAW THE ACTIVE POLYGON ---
                if active_polygon is not None:
                    overlay = frame.copy()
                    
                    # Fill the polygon with bright green
                    cv.fillPoly(overlay, [active_polygon], color=(0, 255, 0))
                    
                    # Draw a white border around it for better visibility
                    cv.polylines(overlay, [active_polygon], isClosed=True, color=(255, 255, 255), thickness=2)
                    
                    # Blend the overlay with the original frame (40% opacity)
                    cv.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

                cv.imshow("Click on a surface", frame)
                if cv.waitKey(1) == ord('q'): 
                    break
    finally:
        print("Safely shutting down...")
        zed.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()