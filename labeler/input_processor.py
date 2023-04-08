from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore

class InputProcessor:
    def __init__(self, log_files_dir: Path, labels_file_path: Path, delimiter: str = ";"):
        self.log_files_dir: Path = log_files_dir
        self.labels_file_path: Path = labels_file_path
        self.delimiter = delimiter

    def _preprocess_log(self, log: str) -> str:
        return log

    def read_logs(self) -> pd.DataFrame:
        labels_data: pd.DataFrame = pd.read_csv(self.labels_file_path, delimiter=self.delimiter, dtype={"path": "string", "labels": "string"}) # type: ignore

        logs: List[str] = []
        for _, row in labels_data.iterrows(): # type: ignore
            log_file: str = row['path'] # type: ignore
            with open(self.log_files_dir.joinpath(log_file), "r") as f: # type: ignore
                log_content: str = f.read()
                log_content = self._preprocess_log(log_content)
                logs.append(log_content)

        data: pd.DataFrame = pd.DataFrame({"log": logs})
        data = pd.concat([data, labels_data], axis=1) # type: ignore

        return data

    def split_data(self, data: pd.DataFrame, test_size_from_full: float = 0.2, val_size_from_remainder: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data: pd.DataFrame
        test_data: pd.DataFrame
        val_data: pd.DataFrame
        train_data, test_data = train_test_split(data, test_size=test_size_from_full, random_state=42) # type: ignore
        train_data, val_data = train_test_split(train_data, test_size=val_size_from_remainder, random_state=42) # type: ignore

        return train_data, val_data, test_data # type: ignore

    def extract_texts_and_labels(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        texts: List[str]
        bug_labels: List[str]
        texts, bug_labels = data["path"].tolist(), data["labels"].tolist()

        return texts, bug_labels
