# QB_methane
Repository to gather the code for Quantum Black challenge on methane leakage.

Model should be implemented using pytorch lightning (see baseline for more details)

To start using the project do the followings:

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

**Launching the Streamlit app**
```python
streamlit run app/app.py
```