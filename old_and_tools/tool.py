import cv2
import pyzed.sl as sl
import numpy as np
import open3d as o3d

# Global variables to handle interaction
accumulated_points = []
clicked_pixel = None

def mouse_callback(event, x, y, flags, param):
    global clicked_pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixel = [x, y]

def main():
    global clicked_pixel, accumulated_points
    
    # Initialize ZED Camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    
    # NEW COORDINATE SYSTEM: Z-UP, Y-FORWARD, X-RIGHT
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        return

    # Enable tracking (Required for native plane detection)
    camera_config = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(camera_config)

    # Window configuration
    window_name = "ZED Boot Calibration (Z-UP)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    runtime_params = sl.RuntimeParameters()
    image_mat = sl.Mat()
    point_cloud_mat = sl.Mat()
    green_overlay_mask = None 

    print("\n=== SYSTEM BOOT: TRAY CALIBRATION (Z-UP COORD SYSTEM) ===")
    print("* Click different segments of the tray to turn them green.")
    print("* Press 'ESC' to clear selection and restart.")
    print("* Press 'S' to stitch the surface and freeze view to inspect.")
    print("* Press 'Q' to force quit.")

    # Get camera calibration parameters for 3D -> 2D projection
    camera_param = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    cx, cy = camera_param.cx, camera_param.cy
    fx, fy = camera_param.fx, camera_param.fy

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)
            
            frame = image_mat.get_data()
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) 
            
            if green_overlay_mask is None:
                green_overlay_mask = np.zeros_like(display_frame)

            # Process a user click
            if clicked_pixel is not None:
                plane = sl.Plane()
                if zed.find_plane_at_hit(clicked_pixel, plane) == sl.ERROR_CODE.SUCCESS:
                    
                    # 1. Get the EXACT 3D boundary points for this chunk
                    bounds = plane.get_bounds()
                    
                    poly_pixels = []
                    for pt in bounds:
                        # Save the raw 3D data for the final stitcher
                        accumulated_points.append([pt[0], pt[1], pt[2]])
                        
                        # 2. Project 3D to 2D pixel space based on RIGHT_HANDED_Z_UP convention:
                        # In Z-UP: X is right, Y is forward depth, Z is vertical height.
                        # Camera projection matrix needs forward depth in the denominator.
                        if pt[1] > 0: # Ensure forward depth (Y) is valid
                            u = int((pt[0] * fx) / pt[1] + cx)
                            v = int((-pt[2] * fy) / pt[1] + cy) # Camera image V-axis goes down, Z-up goes up
                            poly_pixels.append([u, v])
                    
                    # 3. Draw the exact polygon shape onto our green feedback mask
                    if len(poly_pixels) > 2:
                        poly_points_np = np.array(poly_pixels, dtype=np.int32)
                        cv2.fillPoly(green_overlay_mask, [poly_points_np], (0, 255, 0))
                    
                    print(f"Added exact region bounding box! Total points collected: {len(accumulated_points)}")
                else:
                    print("Could not isolate a plane there. Click a less reflective spot.")
                
                clicked_pixel = None

            # Render visual composition
            visual_output = cv2.addWeighted(display_frame, 0.7, green_overlay_mask, 0.3, 0)
            cv2.imshow(window_name, visual_output)
            
            # Key stroke assignments
            key = cv2.waitKey(10) & 0xFF
            
            if key == 27:  # ESC Key triggered
                accumulated_points = []
                green_overlay_mask = np.zeros_like(display_frame)
                print("❌ Selection cleared! Everything reset.")
                
            elif key == ord('s'):  # Stitch and build surface mesh
                if len(accumulated_points) > 5:
                    break
                print("⚠️ Please select a few surface areas before stitching.")
                
            elif key == ord('q'):  # Safety quit
                zed.close()
                cv2.destroyAllWindows()
                return

    # === Process Final Stitched Target Object ===
    if len(accumulated_points) > 5:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(accumulated_points))
        
        # 1. Attempt Alpha Shape
        alpha = 0.08 
        stitched_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        
        # 2. Fallback to Convex Hull if Alpha fails
        if stitched_mesh.is_empty():
            print("\n⚠️ Alpha shape failed. Falling back to Convex Hull...")
            stitched_mesh, _ = pcd.compute_convex_hull()
            print("✅ Convex Hull surface generated.")
        else:
            print(f"\n✅ Alpha Mesh generated with {len(stitched_mesh.triangles)} triangles.")
        
        # --- ROBUST CENTROID APPROACH ---
        vertices = np.asarray(stitched_mesh.vertices)
        triangles = np.asarray(stitched_mesh.triangles)
        
        total_area = 0.0
        area_weighted_center_sum = np.zeros(3)
        
        for tri in triangles:
            p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            tri_center = (p0 + p1 + p2) / 3.0
            tri_area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            
            total_area += tri_area
            area_weighted_center_sum += tri_center * tri_area
            
        if total_area > 0:
            final_center = area_weighted_center_sum / total_area
            print(f"✅ Area-Weighted Centroid calculated over {total_area*10000:.1f} cm²")
        else:
            print("\n⚠️ Mesh has no surface area. Falling back to point cloud average.")
            final_center = pcd.get_center()
        # ---------------------------------
        
        # --- 2D OVERLAY TARGET VISUALIZATION (Z-UP MODIFIED) ---
        if final_center[1] > 0: # Check forward depth (Y) is valid
            u = int((final_center[0] * fx) / final_center[1] + cx)
            v = int((-final_center[2] * fy) / final_center[1] + cy)
            
            cv2.circle(visual_output, (u, v), 12, (0, 0, 255), -1) 
            cv2.circle(visual_output, (u, v), 25, (0, 255, 255), 3) 
            cv2.putText(visual_output, f"CENTROID: X:{final_center[0]:.2f}m Y:{final_center[1]:.2f}m Z:{final_center[2]:.2f}m", 
                        (u + 30, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Modified bounding summaries to respect Z-up directions
            print("\n=== 3D Surface Spatial Summary ===")
            min_bound = stitched_mesh.get_min_bound()
            max_bound = stitched_mesh.get_max_bound()
            print(f"Tray Width    (X Span): {abs(max_bound[0] - min_bound[0])*100:.1f} cm")
            print(f"Tray Distance (Y Span): {abs(max_bound[1] - min_bound[1])*100:.1f} cm")
            print(f"Tray Elevation(Z Span): {abs(max_bound[2] - min_bound[2])*100:.1f} cm")
            
            print(f"\n🚀 CALIBRATION COMPLETE (Centroid Z-Up)")
            print(f"Target Center Coordinates -> X: {final_center[0]:.3f}m, Y: {final_center[1]:.3f}m, Z: {final_center[2]:.3f}m")
            
            print("\nDisplaying confirmation overlay... Press ANY KEY in the video window to finish bootup.")
            cv2.imshow(window_name, visual_output)
            cv2.waitKey(0) 
            
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()