import streamlit as st
import pandas as pd
import folium
from PIL import Image

# Useful functions
def image_location(image_name):
    # Read the metadata.csv file
    metadata_df = pd.read_csv("data/test_data/metadata.csv")

    # Get image id
    image_id = image_name[-8:-4]

    # Filter the metadata for the uploaded image
    image_metadata = metadata_df[metadata_df["id_coord"].str.contains(image_id)]

    # Extract the latitude and longitude values
    latitude = image_metadata["lat"].values[0]
    longitude = image_metadata["lon"].values[0]

    # Create a map centered at the image location
    m = folium.Map(location=[latitude, longitude], zoom_start=10)

    # Add a marker to the map
    folium.Marker([latitude, longitude], popup=image_name).add_to(m)

    return m

def main():
    # Set CSS
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Set sidebar
    with st.sidebar:
        st.image("assets/cleanr.png")

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
        st.sidebar.markdown(
            """
        <div style="text-align: center;">
            <h1 class="sidebar-title">Methane Plume Detector</h1>
            <p class="subtitle">An HEC-QB application</p>
        </div>
        """,
        unsafe_allow_html=True)

        # Set pages options
        page = st.radio("Please select your use case", ["Detect a plume on an image",
                                                        "See current plumes locations"])

    # Methane Plume Detector
    if page == "Detect a plume on an image":
        st.title("🔎 🏭 Detect a plume on an image")
        uploaded_file = st.file_uploader("Please upload your image",
                                 type=["tiff"],
                                 accept_multiple_files=False)
        
        if uploaded_file:
            with st.spinner("Uploading satellite image"):
                ### INSERT MODEL ###
                ### TBU: multiple images, local path, function, color plume ###
                image = Image.open(uploaded_file)
                image = image.convert("RGB")  # Convert to RGB format
                image.save("temp_image.png", "PNG")  # Save the image as PNG
                st.success("Image uploaded!")
            
        col1, col2, col3 = st.columns(3)
            
        # Display the uploaded image
        if col1.button("Display uploaded image"):
            st.image("temp_image.png", caption=uploaded_file.name, use_column_width=True)

        # Display the result
        if col2.button("Check if there is a plume"):
            ### INSERT MODEL RESULTS ###
            st.info("Yes")
        
        # Display the map
        if col3.button("See image location"):
            # Display the map in Streamlit
            #m = image_location(uploaded_file.name)
            #st.markdown("## Image Location on Map")
            #st.write(m, unsafe_allow_html=True)

            # Create a new row of data
            new_row = {'Column1': 4, 'Column2': 'D'}
            #new_row = {date,id_coord,plume,set,lat,lon,coord_x,coord_y,path}
            #df = df.append(new_row, ignore_index=True)

    # Map of the plumes locations
    if page == "See current plumes locations":
        df = pd.read_csv("data/train_data/metadata.csv")
        df['color'] = df['plume'].apply(lambda x: '#FF0000' if x == 'yes' else '#00FF00')
        st.map(df, color='color')
        st.markdown(":green[No methane plume] :red[Methane plume]")


if __name__ == "__main__":
    main()