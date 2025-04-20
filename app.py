import streamlit as st
from PIL import Image

# Set Page Configuration
st.set_page_config(page_title="ECG Arrhythmia Detection", page_icon="💓")

# Load and Display Logo
logo = Image.open("logo.png")  # Ensure logo.png is in the same folder
st.image(logo, width=150)

# Display Title and Developer Name
st.title("ECG Arrhythmia Detection System")
st.subheader("Developed by AKSHAYA RA2311060010011 🎓")

# Navigation Buttons
st.write("### Select an Option Below:")

# Streamlit requires additional pages inside the 'pages/' folder
st.page_link("pages/analysis.py", label="Start Analysis 📊", icon="📈")

