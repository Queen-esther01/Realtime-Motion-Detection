import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Initialize the KNN background subtractor
bg_subtractor = cv2.createBackgroundSubtractorKNN()

# Streamlit app title
st.title("Real-Time Motion Detection")

# Display instructions
st.write("This application captures live video from your webcam and detects motion in real-time.")

# Create a placeholder for the video frames
frame_placeholder = st.empty()

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    st.error("Could not access the webcam. Please check your camera settings.")

try:
    while cap.isOpened():
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam. Exiting...")
            break

        # Resize frame for better performance
        frame_resized = cv2.resize(frame, (640, 480))

        # Step 1 - Apply the background subtractor
        fg_mask = bg_subtractor.apply(frame_resized)

        # STEP 2: erode the mask
        fg_mask_erode = cv2.erode(fg_mask, np.ones((5, 5), np.uint8))
        motion_area = cv2.findNonZero(fg_mask_erode)

        # STEP 3: draw bounding box for motion area
        if motion_area is not None:
            x, y, w, h = cv2.boundingRect(motion_area)
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), thickness=6)

        # Stack original frame and motion mask side by side for comparison
        combined_view = np.hstack((frame, frame_resized))

        # Convert the combined view to a format compatible with Streamlit
        combined_image = Image.fromarray(cv2.cvtColor(combined_view, cv2.COLOR_BGR2RGB))

        # Display the combined image in the Streamlit app
        frame_placeholder.image(combined_image, caption="Original Frame (Left) & Motion Mask (Right)", use_column_width=True)

        # Break the loop if the user stops the app
        key = cv2.waitKey(1)

        if key == ord('Q') or key == ord('q') or key == 27:
            break

finally:
    # Release the video capture and clean up
    cap.release()
    cv2.destroyAllWindows()
