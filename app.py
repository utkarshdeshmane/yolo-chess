import streamlit as st  
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

# Title
st.title("YOLO Object Detection App")
mod=['cards.pt','Chess.pt']

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])   

if uploaded_file is not None:
    # Load image using PIL and convert to NumPy
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Convert RGB to BGR (YOLO and OpenCV use BGR format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load YOLO model
    st.sidebar.title("Model Configuration")
    mod = st.sidebar.selectbox("Select Model", mod)
    model = YOLO(mod)  

    # Get class names from model
    class_names = model.names  

    # Perform inference
    results = model(image_bgr)

    # Extract detections
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs (converted to int)

    # Map class IDs to class names
    class_labels = [class_names[class_id] for class_id in class_ids]

    # Create DataFrame for results
    df = pd.DataFrame(detections, columns=["x1", "y1", "x2", "y2"])
    df["confidence"] = confidences
    df["class_id"] = class_ids
    df["class_name"] = class_labels  # Add class names to the table

    # Display results as a table
    st.write(df)

    # Draw bounding boxes on image
    for i, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        confidence = row["confidence"]
        class_name = row["class_name"]

        # Draw bounding box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{class_name}"
        cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert BGR to RGB before displaying in Streamlit
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Show image with bounding boxes
    st.image(image_rgb, caption="Detected Objects", use_column_width=True)
