import pandas as pd
from labeler.input_processor import InputProcessor


def test_filter_labels_by_count():
    df = pd.DataFrame(
        {
            "text": ["this is an example", "another example", "yet another example"],
            "labels": [["A", "B"], ["A"], ["C", "D"]],
        }
    )

    filtered_df = InputProcessor.filter_labels_by_count(df, 2)
    expected_df = pd.DataFrame({"text": ["this is an example", "another example"], "labels": [["A", "B"], ["A"]]})

    assert expected_df.to_dict() == filtered_df.to_dict()
