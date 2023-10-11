import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
# from ???.py import model

### Useful functions ###

# Get image informations
def image_infos(image_name):
    # Read the metadata.csv file
    metadata_df = pd.read_csv("data/test data/metadata.csv")

    # Get image id
    image_id = image_name[-8:-4]

    # Filter the metadata for the uploaded image
    image_metadata = metadata_df[metadata_df["id_coord"].str.contains(image_id)]

    # Extract the latitude and longitude values
    latitude = image_metadata["lat"].values[0]
    longitude = image_metadata["lon"].values[0]
    location = [latitude, longitude]

    # Extract plume coordinates on picture
    coord_x = image_metadata["coord_x"].values[0]
    coord_y = image_metadata["coord_y"].values[0]
    coords = [coord_x, coord_y]

    return location, coords

# Get satellite view of location
def plot_satellite_image(latitude, longitude):
    fig = plt.figure(figsize=(10, 10))
    m = Basemap(epsg=3857, projection='merc', llcrnrlat=latitude-0.01, urcrnrlat=latitude+0.01, llcrnrlon=longitude-0.01, urcrnrlon=longitude+0.01, resolution='h')
    m.arcgisimage(service='World_Imagery', xpixels=700, verbose=False)
    ### ADD LEGEND WITH COORDINATES ###
    plt.savefig('app/satellite_image.png')  # Save the plotted image

# Create session state elements
if 'image' not in st.session_state:
    st.session_state.image = None
if 'loc' not in st.session_state:
    st.session_state.loc = None
if 'coords' not in st.session_state:
    st.session_state.coords = None


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
        st.sidebar.markdown(
            """
        <div style="text-align: center;">
            <h1 class="sidebar-title">Methane Plume Detector</h1>
            <p class="subtitle">An HEC-QB application</p>
        </div>
        """,
        unsafe_allow_html=True)
        if st.button("Analyze a new image"):
            st.session_state.image = None
            st.session_state.loc = None
            st.session_state.coords = None

    # Methane Plume Detector
    st.title("🔎 🏭 Detect a methane plume on an image")
    uploaded_file = st.file_uploader("Please upload your image and click on 'Analyze image'",
                                        type=["tiff"],
                                        accept_multiple_files=False)
        
    if st.button('Analyze image'):
        with st.spinner("Uploading satellite image"):
            # Get image informations (location and coordinates)
            st.session_state.loc, st.session_state.coords = image_infos(uploaded_file.name)

            # Get satellite image
            if st.session_state.image == None:
                plot_satellite_image(st.session_state.loc[0], st.session_state.loc[1])
                st.session_state.image = "app/satellite_image.png"

            ### INSERT MODEL ###
            # output = model.predict(uploaded_file.name)
            st.success("Image uploaded!")

    if st.session_state.image != None:
        display = st.radio("Select what you want to do", ["Display satellite view",
                                                            "Search methane plume",
                                                            "See plumes map"], horizontal=True)
        
        # Display the uploaded image
        if display == "Display satellite view":
            st.image(st.session_state.image, caption=uploaded_file.name, use_column_width=True)

        # Display the result
        if display == "Search methane plume":
            ### INSERT MODEL RESULTS ###
            # st.info(output)
            st.info("Yes")
            
        # Display the map
        if display == "See plumes map":
            # Add color to dataset
            df = pd.read_csv("data/train data/metadata.csv")
            df['color'] = df['plume'].apply(lambda x: '#FF0000' if x == 'yes' else '#00FF00')

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

            st.map(df, color='color', zoom=1)
            st.info("##### Legend:" + "\n"
            +"- :green[No methane plume]" + "\n"
            + "- :red[Methane plume]" + "\n"
            + "- :blue[Current plume under analysis]")


if __name__ == "__main__":
    main()