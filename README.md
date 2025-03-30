# pycraft-training-credit-default-prediction

## Exercice 1

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
| dvc          | >=3.54.1,<3.55 |
| dvclive      | >=3.48.0       |
| xgboost      | >=2.1.1,<2.2   |

To learn more about how to write a `pyproject.toml` file [here](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
