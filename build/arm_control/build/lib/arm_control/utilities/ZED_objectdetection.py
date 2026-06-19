#!/usr/bin/env python3
import sys
import pyzed.sl as sl
import cv2 as cv
import numpy as np

def detect_and_draw_objects(frame, objects):
    """
    Generic helper function to draw 2D bounding boxes and labels 
    for detected objects onto the OpenCV frame.
    """
    for obj in objects.object_list:
        # Extract the 2D bounding box (returns 4 corners: TL, TR, BR, BL)
        bbox = obj.bounding_box_2d
        
        # Ensure the bounding box is valid
        if bbox is not None and len(bbox) == 4:
            # Grab Top-Left and Bottom-Right corners for OpenCV rectangle
            pt1 = (int(bbox[0][0]), int(bbox[0][1]))
            pt2 = (int(bbox[2][0]), int(bbox[2][1]))
            
            # Draw the bounding box (Orange color)
            cv.rectangle(frame, pt1, pt2, (0, 165, 255), 2)
            
            # Format the label (Strip the "OBJECT_CLASS." prefix for clean text)
            main_label = str(obj.label).replace("OBJECT_CLASS.", "")
            sub_label = str(obj.sublabel).replace("OBJECT_SUBCLASS.", "")
            
            # If there's a specific sub-label (like 'sneaker' under 'clothing'), include it
            if "UNCLASS" not in sub_label:
                display_text = f"{main_label} ({sub_label}) {int(obj.confidence)}%"
            else:
                display_text = f"{main_label} {int(obj.confidence)}%"
                
            # Draw a background filled rect for the text so it's readable
            (text_width, text_height), _ = cv.getTextSize(display_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(frame, (pt1[0], pt1[1] - text_height - 10), (pt1[0] + text_width, pt1[1]), (0, 165, 255), -1)
            
            # Draw the text overlay
            cv.putText(frame, display_text, (pt1[0], pt1[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame

def main():
    print("Initializing ZED Camera for Object Detection...")
    
    zed = sl.Camera()
    init = sl.InitParameters()
    
    # Keeping resolution/fps low to prevent CPU explosions
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("CRITICAL: Failed to open ZED camera interface!")
        sys.exit(1)

    # 1. Enable Positional Tracking (Required for object tracking)
    zed.enable_positional_tracking(sl.PositionalTrackingParameters())
    
    # 2. Configure and Enable Object Detection
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True # Tracks objects seamlessly across frames
    obj_param.enable_segmentation = False # We just want boxes, not full masks
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE
    
    if zed.enable_object_detection(obj_param) != sl.ERROR_CODE.SUCCESS:
        print("CRITICAL: Failed to enable Object Detection Module!")
        zed.close()
        sys.exit(1)

    # 3. Configure Runtime Parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40 # Adjust sensitivity here (0-100)
    
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    objects = sl.Objects() # Container to hold the AI outputs
    
    print("Successfully connected. Press 'q' to exit.")

    try:
        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                # Retrieve visual frame
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()
                
                # Strip alpha channel for clean OpenCV usage
                if frame.shape[2] == 4:
                    frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
                
                # Retrieve AI Object Data
                zed.retrieve_objects(objects, obj_runtime_param)
                
                # --- INJECT HELPER FUNCTION ---
                frame = detect_and_draw_objects(frame, objects)

                cv.imshow("ZED Object Detection Feed", frame)
                if cv.waitKey(1) == ord('q'): 
                    break
    finally:
        print("Safely shutting down AI modules and camera...")
        zed.disable_object_detection()
        zed.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()