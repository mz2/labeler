from pathlib import Path
from pytest import approx  # type: ignore

import pandas as pd
import pytest

from labeler.input_processor import InputProcessor
from transformers import AutoTokenizer  # type: ignore


def setup_module():
    global processor, log_files_dir, labels_file_path, delimiter
    log_files_dir = Path("tests/fixtures/doctype")
    labels_file_path = log_files_dir / "labels.csv"
    delimiter = ";"
    processor = InputProcessor(log_files_dir, labels_file_path, delimiter)


@pytest.mark.unit
def test_init():
    assert processor.log_files_dir == log_files_dir
    assert processor.labels_file_path == labels_file_path
    assert processor.csv_delimiter == delimiter


@pytest.mark.unit
def test_preprocess_log():
    log = "Sample log content"
    assert processor._preprocess_log(log) == log  # type: ignore


@pytest.mark.unit
def test_read_logs():
    data, labels, labels_to_index, index_to_labels = processor.read_data(
        tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")
    )
    assert isinstance(data, pd.DataFrame)
    assert labels == ["html", "json", "md", "yaml"]
    assert sorted(labels_to_index.keys()) == labels
    assert sorted(index_to_labels.keys()) == [0, 1, 2, 3]

    assert len(data) == len(pd.read_csv(labels_file_path, delimiter=delimiter))  # type: ignore


@pytest.mark.unit
def test_split_data():
    data, _, _, _ = processor.read_data(tokenizer=AutoTokenizer.from_pretrained("bert-base-cased"))
    train_data, val_data, test_data = processor.split_data(data, test_size_from_full=0.2, val_size_from_remainder=0.25)

    assert (len(train_data) / len(data)) == approx(0.6, abs=0.02)
    assert (len(val_data) / len(data)) == approx(0.2, abs=0.02)
    assert (len(test_data) / len(data)) == approx(0.2, abs=0.02)
