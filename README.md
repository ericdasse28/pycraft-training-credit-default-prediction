# pycraft-training-credit-default-prediction

## Exercise 1: Migration to Poetry for dependency management

1. Install [Poetry](https://python-poetry.org/docs/1.8#installing-with-pipx)
2. In [pyproject.toml](./pyproject.toml), rewrite the dev dependency group as follows:

   ```toml
   [tool.poetry.group.dev.dependencies]
   pytest = "^8.3.5"
   ```

   This ensures Poetry can read the development dependency group

3. Lock the current project dependencies using the command: `poetry lock`
4. Create a virtual environment named `.venv` at the project root (hats off if you did already!)
5. Install **all** the dependencies (main and dev) in that environment

```shell
poetry install
```

## Exercise 2: Formatting and linting

### Formatting with Black & isort

1. Specify and install `black` and `isort` as dev dependencies group as follows:

```shell
poetry add -G dev black isort
```

2. Is the codebase currently formatted according to Black style? If not, apply Black formatting on it
3. Are the imports sorted currently sorted correctly?
4. Configure your VS Code so that Black automatically formats Python files on save
   > **Tip**: Search the _Black Formatter_ extension in VS Code extensions tab
5. Likewise, configure your VS Code so it uses isort.

### Linting with Flake8

1. Specify and install `flake8` as a dev dependency
2. Run linting on the entire codebase. Notice anything weird?
3. Create a file named `setup.cfg` at the root of the project and add the following:

```Properties
[flake8]
exclude =
    .venv,
    __pycache__
```

This configuration tells Flake8 to ignore any file that's within your virtual environment
or in `__pycache__`. 4. Run linting again. Did it detect any problem? 5. Add the Flake8 extension to your VS Code

### Bonus: sharing recommended VS Code extensions across team

It is possible to make it so VS Code prompts users to install the exact
extensions we are using in the project within their workspace. It can be done
by creating a file named `extensions.json`. Let's make one and add it to version
control.

## Exercise 3: refactoring

1. Open the data preprocessing module: [credit_default_prediction/data_preprocessing.py](./credit_default_prediction/data_preprocessing.py)
2. What do you notice?
3. Refactor the `preprocess` function until it becomes more readable, and its McCabe complexity drops below 5.

   The following Pandas resources will be of help:

- [pandas.DataFrame.dropna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)
- [pandas.Series.clip](https://pandas.pydata.org/docs/reference/api/pandas.Series.clip.html#pandas.Series.clip)
- [pandas.Series.map](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html#pandas.Series.map)
- [pandas.Series.fillna](https://pandas.pydata.org/docs/reference/api/pandas.Series.fillna.html#pandas.Series.fillna)
