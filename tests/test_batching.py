import pytest
from labeler.batch_data import batch_data


@pytest.fixture
def sample_text():
    return """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10"""


def test_batch_data_without_max_bytes_nor_overlap(sample_text: str):
    size = 4
    overlap = 0
    expected_output = ["Line 1\nLine 2\nLine 3\nLine 4", "Line 5\nLine 6\nLine 7\nLine 8", "Line 9\nLine 10"]

    assert batch_data(sample_text, size, overlap) == expected_output


def test_batch_data_without_max_bytes(sample_text: str):
    size = 4
    overlap = 1
    expected_output = [
        "Line 1\nLine 2\nLine 3\nLine 4",
        "Line 4\nLine 5\nLine 6\nLine 7",
        "Line 7\nLine 8\nLine 9\nLine 10",
        "Line 10",
    ]

    assert batch_data(sample_text, size, overlap) == expected_output


def test_batch_data_with_just_max_bytes(sample_text: str):
    size = 4
    overlap = 0
    max_bytes = 6
    expected_output = ["Line 1", "Line 5", "Line 9"]

    assert batch_data(sample_text, size=size, overlap=overlap, max_bytes=max_bytes) == expected_output


def test_batch_data_with_overlap_and_max_bytes(sample_text: str):
    size = 4
    overlap = 1
    max_bytes = 20
    expected_output = ["Line 1\nLine 2\nLine 3", "Line 4\nLine 5\nLine 6", "Line 7\nLine 8\nLine 9", "Line 10"]
    assert batch_data(sample_text, size, overlap, max_bytes) == expected_output
