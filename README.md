# pycraft-training-credit-default-prediction

## Exercise 1: Testing the preprocessing pipeline

The preprocessing pipeline is not tested at all, which highly impeded
maintainability.

Write a test that represents the expected behavior of the pipeline.

## Exercise 2: Save dataset

We have a module dedicated to handling datasets operations named [dataset.py](./src/credit_default_prediction/dataset.py). Implement a function that can save a dataset object to a specified file path with a TDD approach

## Exercise 3: Incorporating Kedro

For the following, do not hesitate to check [Kedro documentation](https://docs.kedro.org/en/stable/)

### Turn project into Kedro project

Using [this documentation](https://docs.kedro.org/en/stable/get_started/minimal_kedro_project.html),
as well as everything we have learnt so far,do _just enough_ to turn this project into a Kedro project.
