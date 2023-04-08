import os
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

class LogProcessor:
    def __init__(self, log_files_dir: str, labels_file_path: str):
        self.log_files_dir = log_files_dir
        self.labels_file_path = labels_file_path

    def __preprocess_log(self, log: str) -> str:
        return log

    def read_logs(self) -> pd.DataFrame:
        log_files: List[str] = os.listdir(self.log_files_dir)

        logs: List[str] = []
        for log_file in log_files:
            with open(os.path.join(self.log_files_dir, log_file), "r") as f:
                log_content: str = f.read()
                log_content = self.__preprocess_log(log_content)
                logs.append(log_content)

        labels_data: pd.DataFrame = pd.read_csv(self.labels_file_path)

        data: pd.DataFrame = pd.DataFrame({"log": logs})
        data = pd.merge(data, labels_data, left_index=True, right_index=True)

        return data

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data: pd.DataFrame
        test_data: pd.DataFrame
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

        return train_data, val_data, test_data

    def extract_texts_and_labels(self, data: pd.DataFrame) -> Tuple[List[str], List[int]]:
        texts: List[str]
        bug_labels: List[int]
        texts, bug_labels = data["log"].tolist(), data["bug_labels"].tolist()

        return texts, bug_labels
