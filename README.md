# Methane MRV Study

CleanR satellite images and deep learning to provide a method for monitoring, reporting and verification (MRV) of methane emissions for reporting.

Project: localize methane leaks in the atmosphere.

## Project Setup

- Create a virtual environmnent and activate it

```bash
conda create --name qb_env python=3.8
conda activate qb_env
```

- Install the necessary requirements and the project package

```python
pip install -r requirements.txt
pip install -e .
```

- Unzip data inside the data folder

- Run a cross validation for the baseline

```bash
python main.py
```

**Linting and formating**

- Do not forget to use Black formatter before pushing to avoid linting conflicts
- Please use isort extension to sort your imports

## Model Development

### Data: 

Satellite images [data set – 64 x 64 images in greyscale] of different locations.
* path
* date the satelite image was taken
* class (`plume` or `no_plume`)
* an ID identifying the location
* latitude and longitude coordinates locating the center of the plume (`lat`,`lon`)
* pixel coordinates locating the center of the plume in the image (`coord_x`,`coord_y`). Please be midnful that the axis origin (0,0) is at the top left corner of the image

The dataset contains two folders:
- `plume` : contains all images with plumes of methane.
- `no_plume` : contains all images with no plume of methane.

### Target

Identify whether each location contains a methane plume or not.

### Model

Model should be implemented using pytorch lightning (see baseline for more details)

## Business Plan

Identify use cases where this model can be used to drive positive impact. 
**Launching the Streamlit app**
```python
streamlit run app/app.py
```
