**QB Methane MRV Study**

Methane is a powerful greenhouse gas, with a global warming potential many times greater than carbon dioxide. Monitoring, Reporting, and Verification (MRV) of methane emissions is crucial in the context of climate change mitigation and environmental protection.

In this project, we are aiming to leverage advanced deep learning techniques and satellite imagery to develop an efficient and accurate system for localizing methane leaks in the atmosphere.

**Dataset Details**:

The satellite imagery dataset is greyscale and consists of 64x64 pixel images, each associated with:

- Path
- Date of capture
- Class (either `plume` indicating presence of methane or `no_plume` indicating absence)
- A unique ID for the location
- Latitude and longitude coordinates (lat,lon) of the center of the plume
- Pixel coordinates (coord_x, coord_y) locating the center of the plume in the image (Note: the origin (0,0) is at the top left corner of the image)

The dataset is divided into two folders:

- `plume`: Contains images depicting methane plumes.
- `no_plume`: Contains images without methane plumes.

**Setting Up the Project**:

```bash
# Create a virtual environment and activate it
conda create --name qb_env python=3.8
conda activate qb_env

# Install necessary requirements and the project package
pip install -r requirements.txt
pip install -e .

# Unzip the dataset inside the data folder and store with .gitkeep files

# Execute the baseline model
python main.py
```

**Linting & Formatting**:

- Before pushing any changes, use the Black formatter to avoid linting conflicts.
- Ensure your imports are organized using the isort extension.

**Model Development**:

For this project, the model will be implemented using Pytorch Lightning. A sample baseline has been provided for reference.

Inspired by the research article "Gas Classification Using Deep Convolutional Neural Networks", we aim implement a Deep Convolutional Neural Network (DCNN) known as GasNet for classifying the satellite images. Additionally, we explor the capabilities of GasNet 2, which is known from the research "Machine Vision for Natural Gas Methane Emissions Detection Using an Infrared Camera".

**Potential Impact & Use Cases**:

The successful implementation of this model can revolutionize methane leak detection in various industries. Our application is build for the **Oil and Gas**: industry. Our model could impact:

1. Oil & Gas: Rapid detection of leaks in pipelines and storage units.
2. Agriculture: Identifying methane emissions from livestock and agricultural processes.
3. Waste Management: Monitoring landfills and waste treatment facilities for methane emissions.

**Launching the Application**:

To visualize the model's outputs and get insights into its performance, launch the Streamlit app:

```python
streamlit run app/app.py
```