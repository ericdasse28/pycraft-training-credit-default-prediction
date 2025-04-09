# pycraft-training-credit-default-prediction

## Exercise 1: Dependency management, linting and formatting with Poetry and Ruff

1. Install [Poetry](https://python-poetry.org/docs/1.8#installing-with-pipx)
2. In [pyproject.toml](./pyproject.toml), rewrite the dev dependency group as follows:

   ```toml
   [tool.poetry.group.dev.dependencies]
   pytest = "^8.3.5"
   pytest-mock = "^3.14.0"
   ```

   This ensures Poetry can read the development dependency group

3. Lock the current project dependencies using the command: `poetry lock`
4. Create a virtual environment (if you didn't already)
5. Install all the dependencies (main and dev) in that environment
6. Add `ruff` to dev dependency group using: `poetry add -G dev ruff`
7. Is code format okay for Ruff? If not, apply ruff formatting on the entire codebase
8. Is linting okay? If not, fix any issue you may encounter
9. Configure your VS Code so that Ruff automatically formats Python files on save
10. Run `pytest` to make sure all tests pass
