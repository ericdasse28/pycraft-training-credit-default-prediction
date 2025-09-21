# pycraft-training-credit-default-prediction

## Exercise 1: Writing a `pyproject.toml`

With help of the documentation [here](https://packaging.python.org/en/latest/guides/writing-pyproject-toml), complete this exercice:

1. Create a pyproject.toml file
2. Declare `setuptools` as your build backend. Learn how [here](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#declaring-the-build-backend)
3. Declare basic project metadata:

```toml
[project]
name = "credit-default-prediction"
version = "0.1.0"
```

4. Specify project dependencies:
   In this part, you will list the dependencies below as part of your project and put a constraint on the versions as described:

| Dependency   | Version        |
| ------------ | -------------- |
| scikit-learn | >=1.5.1,<1.6   |
| xgboost      | >=2.1.1,<2.2   |
| pandas       | >=2.2.3,<2.3   |

5. Add `pytest` as an optional dependency in a group named `dev`

## Exercise 2: Locking pyproject.toml dependencies with pip-tools

1. Create a virtual environment and subsequently activate it

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install pip-tools

```bash
python -m pip install pip-tools
```

3. Lock project dependencies and generate a requirements.txt from it

```bash
pip-compile -o requirements.txt pyproject.toml
```

4. Lock development dependencies

```bash
pip-compile --extra dev -o dev-requirements.txt pyproject.toml
```

5. Look inside requirements.txt, what do you notice?
6. Same question for dev-requirements.txt

## Exercise 3: Installing locked dependencies

Install both project and dev dependencies

```bash
pip-sync requirements.txt dev-requirements.txt
```
