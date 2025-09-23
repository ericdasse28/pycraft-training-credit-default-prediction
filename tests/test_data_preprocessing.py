import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from credit_default_prediction.data_preprocessing import preprocess


def test_preprocess():
    original_loan_data = pd.DataFrame(
        {
            "loan_int_rate": [3.5, 1.2, np.nan, 0.80],
            "person_emp_length": [70, 12, 19, 2],
            "person_income": [13, 1_000_000_000_000, 2500, 9500],
            "person_education": [
                "Secondary",
                "POSTGRAD",
                "Pas ton problème",
                "GRADUATE",
            ],
            "cb_person_default_on_file": ["Y", "N", "N", "Y"],
        }
    )

    actual_loan_data = preprocess(original_loan_data)

    expected_loan_data = pd.DataFrame(
        {
            "loan_int_rate": [1.2, 0.80],
            "person_emp_length": [12, 2],
            "person_income": [1_000_000, 9500],
            "person_education": [
                "Post Graduate",
                "Graduate",
            ],
            "cb_person_default_on_file": [0, 1],
        },
        index=[1, 3],
    )
    assert_frame_equal(expected_loan_data, actual_loan_data)
