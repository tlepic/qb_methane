# QB_methane
Repository to gather the code for Quantum Black challenge on methane leakage.

Model should be implemented using pytorch lightning (see baseline for more details)

To start using the project do the followings:

- Create a virtual environmnent and activate it

```bash
$conda create env -name qb_env -python=3.8
$conda activate qb_env
```

- Install the necessary requirements and the project package

```python
pip install -r requirements.txt
pip install -e .
```

- Run a cross validation for the baseline

```bash
$python main.py
```