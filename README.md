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

## Exercise 3: Saving trained model to a model registry

In [training_pipeline.py](./credit_default_prediction/training_pipeline.py), we defined a function
that trains a model on input data and returns the trained model.

To properly manage our models lifecycle, we decided that our training pipeline must save the model it trained to a model registry.

Implement that feature with a TDD approach.

## Exercise 4: Model evaluation

Once models are trained, they need to be evaluated. To perform evaluation,
we first need to retrieve the trained model from the model registry. Enable that feature with a TDD approach.

## Exercise 5: Concrete MLFlow model registry

Implement an **actual** model registry using the `mlflow` distribution package.

## Bonus: Create a Poetry script to facilitate training pipeline execution

Using Poetry `scripts` feature, create a command named `train` that runs the training
pipeline command **defined** in [training_pipeline.py](./credit_default_prediction/training_pipeline.py)

## Bonus: Add missing tests

Write tests for any important aspect of the project that is not tested
