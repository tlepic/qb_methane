from methane.models import TestModel
import torch
import tifffile as tiff
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Ellipse

model = TestModel()
model = model.load_from_checkpoint("app/tools/model_checkpoint.ckpt")
model.eval()

# Get methane plume prediction
def methane_predict(image, coordinates):
    """
    Predicts the presence of a methane plume in an image and saves the image with a highlighted ellipse if a plume is detected.

    Args:
        image (str): The name of the image file.
        coordinates (tuple): The coordinates (x, y) of the plume on the image.

    Returns:
        float: The predicted probability of a methane plume being present.
    """
    # Converting the image into a tensor
    with tiff.TiffFile("data/test_data/images/"+image) as tif:
        _feature = tif.asarray().astype(np.float64)
    X = torch.tensor(_feature)
    X = X.reshape(1, 1, 64, 64)

    # Calculating the output
    output = F.sigmoid(model(X.float()))

    # Saving the image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(_feature, cmap='gray')
    ax.axis('off')

    if round(output.item()) == 1:
        # Draw ellipse
        ellipse = Ellipse(xy=coordinates, width=10, height=20, edgecolor='r', facecolor='none')
        ax.add_patch(ellipse)
    plt.savefig('app/tmp/temp_img.png', bbox_inches='tight', pad_inches=0)

    return output.item()


# Get image informations
def image_infos(image_name):
    """
    Retrieves the location and plume coordinates associated with an image.

    Args:
        image_name (str): The name of the image file.

    Returns:
        tuple: A tuple containing the location (latitude, longitude) and plume coordinates (x, y) on the image.
    """
    # Read the metadata.csv file
    metadata_df = pd.read_csv("data/test_data/metadata.csv")

    # Get image id
    image_id = image_name[-8:-4]

    # Filter the metadata for the uploaded image
    image_metadata = metadata_df[metadata_df["id_coord"].str.contains(image_id)]

    # Extract the latitude and longitude values
    latitude = image_metadata["lat"].values[0]
    longitude = image_metadata["lon"].values[0]
    location = (latitude, longitude)

    # Extract plume coordinates on picture
    coord_x = image_metadata["coord_x"].values[0]
    coord_y = image_metadata["coord_y"].values[0]
    coords = (coord_x, coord_y)

    return location, coords


# Get satellite view of location
def plot_satellite_image(latitude, longitude):
    """
    Retrieves and saves a satellite image of a given location.

    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.

    Returns:
        str: The path to the saved satellite image.
    """
    # Get the right satellite image
    fig = plt.figure(figsize=(10, 10))
    m = Basemap(epsg=3857, projection='merc', llcrnrlat=latitude-0.005, urcrnrlat=latitude+0.005, llcrnrlon=longitude-0.005, urcrnrlon=longitude+0.005, resolution='h')
    m.arcgisimage(service='World_Imagery', xpixels=1000, verbose=False)
    
    # Save satellite image
    plt.savefig('app/tmp/satellite_image.png', bbox_inches='tight', pad_inches=0)

    return 'app/tmp/satellite_image.png'