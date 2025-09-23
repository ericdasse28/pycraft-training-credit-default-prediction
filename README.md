# pycraft-training-credit-default-prediction

## Exercise 1: Testing the preprocessing pipeline

The preprocessing pipeline is not tested at all, which highly impeded
maintainability.

Write a test that represents the expected behavior of the pipeline.

## Exercise 2: Save dataset

We need a method to save pre-processed dataset to a CSV file. To accomplish that,
we decided that we need a function with the following behavior:

1. Given a valid dataframe and a file path, said dataframe is saved to the file path
   as CSV file.
2. A dataframe is valid if its columns are
   `person_age`, `loan_int_rate`, `person_emp_length`, `person_income`, `person_education`, `cb_person_default_on_file`, `loan_status`,
   in that order. If a dataframe is not valid, raise a custom exception named `InvalidLoanColumnsException`

Implement these behaviors with a TDD approach.

## Exercise 3: Incorporating Kedro

For the following, do not hesitate to check [Kedro documentation](https://docs.kedro.org/en/stable/)

### Turn project into Kedro project

Using [this documentation](https://docs.kedro.org/en/stable/get_started/minimal_kedro_project.html),
as well as everything we have learnt so far,do _just enough_ to turn this project into a Kedro project.
