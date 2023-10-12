import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from tools.utils import methane_predict, image_infos, plot_satellite_image

# Initialize session state elements
if 'sat_image' not in st.session_state:
    st.session_state.sat_image = None
if 'loc' not in st.session_state:
    st.session_state.loc = None
if 'coords' not in st.session_state:
    st.session_state.coords = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

def main():
    # Set CSS
    with open("app/assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Set sidebar
    with st.sidebar:
        st.image("app/assets/cleanr.png")

        # Text markdown settings
        st.sidebar.markdown(
        """
        <style>
        .sidebar-title {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 10px; /* Ajustez la hauteur en fonction de vos besoins */
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True)

        # Title of the app
        st.sidebar.markdown(
            """
        <div style="text-align: center;">
            <h1 class="sidebar-title">🌱 GREENOPS</h1>
            <p class="subtitle">An HEC-QB application</p>
        </div>
        """,
        unsafe_allow_html=True)

        # Add application logo
        st.image("app/assets/greenops.png")
        
        # App features
        feature = st.radio("Please select your use case", ["🏭 Methane plume detection",
                                                            "🌍 Ecological impact evaluation",
                                                            "🔧 Predictive maintenance",
                                                            "📊 Energy efficiency analysis",
                                                            "📋 Compliance monitoring",
                                                            "🤖 Get advice from GreenBot"])


    # Methane Plume Detector
    if feature == "🏭 Methane plume detection":
        st.title("🔎 🏭 :green[Methane Plume Detector]")
        uploaded_file = st.file_uploader("Please upload your image and click on 'Analyze image'",
                                            type=["tiff"],
                                            accept_multiple_files=False)

        if st.button('Upload image'):
            with st.spinner("Image analysis in progress..."):
                # Create tmp folder to store images
                if not os.path.exists("app/tmp"):
                    os.makedirs("app/tmp")

                # Get image informations (location and coordinates)
                st.session_state.loc, st.session_state.coords = image_infos(uploaded_file.name)

                # Get satellite image
                st.session_state.sat_image = plot_satellite_image(st.session_state.loc[0],
                                                                        st.session_state.loc[1])

                # Get prediction
                st.session_state.prediction = methane_predict(uploaded_file.name,
                                                                st.session_state.coords)

        if st.session_state.sat_image != None:
            display = st.radio("Select what you want to do", ["📊 See image analysis",
                                                                "🗺️ See image location"])
            
            # Display the image analysis
            if display == "📊 See image analysis":
                # Display prediction
                if round(st.session_state.prediction) == 1:
                    st.error("# ⚠️ Methane plume detected!")
                    # Add guidelines
                    if st.checkbox("See guidelines"):
                        st.info("##### Guidelines" + "\n"
                                + "1. :red[Immediate action]" + "\n"
                                + "2. [Safety measures](https://www.google.com)" + "\n"
                                + "3. Isolate the source" + "\n"
                                + "4. [Inform relevant authorities](https://urlz.fr/nZPK)" + "\n"
                                + "5. Investigate the root cause" + "\n"
                                + "6. Mitigation measures" + "\n")
                else:
                    st.success("# 👍 No methane plume detected")

                # Display uploaded image
                try:
                    st.image("app/tmp/temp_img.png",
                                caption=uploaded_file.name,
                                use_column_width=True)
                except:
                    st.image("app/tmp/temp_img.png")

            # Display the image location
            if display == "🗺️ See image location":
                # Add color to dataset
                df = pd.read_csv("data/train_data/metadata.csv")
                df['color'] = df['plume'].apply(lambda x: '#FF0000' if x == 'yes' else '#00FF00')

                # Add new row with image info to dataset
                new_row = pd.DataFrame({
                    'date': [" "],
                    'id_coord': [" "],
                    'plume': [" "],
                    'set': [" "],
                    'lat': [st.session_state.loc[0]],
                    'lon': [st.session_state.loc[1]],
                    'coord_x': [st.session_state.coords[0]],
                    'coord_y': [st.session_state.coords[1]],
                    'path': [" "],
                    'color': ['#0000FF']
                })
                df = pd.concat([df, new_row], ignore_index=True)

                # Display map of plumes
                st.info("#### Map of my plants" +"\n"
                +"*Legend: :green[No methane plume], :red[Methane plume], :blue[Current plume under analysis]* ")
                st.map(df, color='color', zoom=1)

                # Display satellite image
                st.info("#### Satellite view of analyzed plant")
                st.image(st.session_state.sat_image,
                            caption=f"Latitude: {st.session_state.loc[0]} - Longitude: {st.session_state.loc[1]}",
                            use_column_width=True)
    
    # Unavailable features
    if feature in ["🌍 Ecological impact evaluation",
                    "🔧 Predictive maintenance",
                    "📊 Energy efficiency analysis",
                    "📋 Compliance monitoring"]:
        st.error("# ⚙️ Feature in maintenance")

    # GreenBot
    if feature == "🤖 Get advice from GreenBot":
        st.title("🤖 :green[Chat with GreenBot]")
        input_container = st.container()
        response_container = st.container()
        with input_container:
            user_input = st.text_input("Ask GreenBot for advice")
        with response_container:
            with st.chat_message("user"):
                st.markdown("Hello")
            with st.chat_message("assistant"):
                st.markdown("Hello, I am GreenBot, your AI assistant, how can I help you?")


if __name__ == "__main__":
    main()