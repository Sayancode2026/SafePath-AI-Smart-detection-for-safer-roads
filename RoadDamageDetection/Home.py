import streamlit as st

st.set_page_config(
    page_title="SafePath AI – Smart detection for safer roads.",
    page_icon="🛣️",
)

st.image("./resource/banner1.jpg", use_column_width="always")
st.divider()
st.title("SafePath AI – Smart detection for safer roads.")

st.markdown(
     """"Powered by YOLOv8 | Trained on Custom Datasets.

Enhancing road safety and infrastructure reliability through intelligent detection.

Our Road Damage Detection app leverages the power of the YOLOv8 deep learning model, meticulously trained on the Custom  dataset. The model specializes in the rapid identification and classification of common road surface damages to assist in proactive road maintenance and hazard prevention.

🛣️ What Can It Detect?
The system accurately detects and classifies four major types of road damage:

Longitudinal Cracks

Transverse Cracks

Alligator Cracks

Potholes

🧠 Model Insights
Model: YOLOv8 (Small Variant)

Dataset: Custom Dataset — comprising annotated road images.

📸 Input Options
Choose from multiple modes of detection based on your use case:

Real-time Webcam

Pre-recorded Video

Static Images

Easily switch between these input types using the sidebar to test and deploy in different scenarios.

📚 Documentation & Contact
For further details, feature requests, or collaboration: 
📧 Email: sayanbardhan2022@iem.edu.in

🔖 License & Acknowledgements
Dataset: Custom Datasets from Roboflow.

Model Framework: YOLOv8 by Ultralytics

Frontend: Streamlit

All tools and datasets used comply with their respective licenses. This project is built with the intent to contribute to smarter cities and safer roads."""

)

st.divider()

st.markdown(
    """
    This project is created for the Completion of the mandatory final year project of Btech. by Sayan Bardhan for Evaluation.
    """
    
)
