from pathlib import Path
from typing import List, Tuple, Dict

from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split  # type: ignore
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore


class InputProcessor:
    def __init__(self, log_files_dir: Path, labels_file_path: Path, delimiter: str = ";", label_delimiter: str = " "):
        self.log_files_dir: Path = log_files_dir
        self.labels_file_path: Path = labels_file_path
        self.csv_delimiter = delimiter
        self.label_delimiter = label_delimiter

    def _preprocess_log(self, log: str) -> str:
        return log

    @staticmethod
    def convert_labels_to_binary(
        labels_list: List[str], sorted_labels: List[str], label_to_index: Dict[str, int]
    ) -> List[float]:
        binary_encoding = [0.0] * len(sorted_labels)
        for label in labels_list:
            index = label_to_index[label]
            binary_encoding[index] = 1.0
        return binary_encoding

    def sorted_labels(self, data: DataFrame) -> List[str]:
        all_labels = []
        for _, row in data.iterrows():
            all_labels.extend(row["labels"].split(self.label_delimiter))
        return sorted(list(set(all_labels)))

    def read_data(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    ) -> Tuple[DataFrame, List[str], Dict[str, int], Dict[int, str]]:
        labels_data = read_csv(
            self.labels_file_path, delimiter=self.csv_delimiter, dtype={"path": "string", "labels": "string"}
        )

        sorted_labels = self.sorted_labels(labels_data)
        label_to_index = {label: i for i, label in enumerate(sorted_labels)}
        index_to_label = {i: label for i, label in enumerate(sorted_labels)}

        texts: List[str] = []
        labels: List[List[float]] = []
        input_ids = []
        token_type_ids = []
        attention_masks = []

        for _, row in labels_data.iterrows():
            path: str = row["path"]
            with open(self.log_files_dir.joinpath(path), "r") as f:
                content = f.read()
                content = self._preprocess_log(content)
                texts.append(content)
                label_list = InputProcessor.convert_labels_to_binary(
                    row["labels"].split(" "), sorted_labels=sorted_labels, label_to_index=label_to_index
                )
                labels.append(label_list)

                tokenized_input = tokenizer.encode_plus(
                    content,  # type: ignore
                    truncation=True,  # type: ignore
                    padding=True,  # type: ignore
                    add_special_tokens=True,
                )

                input_ids.append(tokenized_input["input_ids"])
                attention_masks.append(tokenized_input["attention_mask"])
                token_type_ids.append(tokenized_input["token_type_ids"])

        return (
            DataFrame(
                {
                    "text": texts,
                    "path": labels_data["path"],
                    "labels": labels,
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_masks": attention_masks,
                }
            ),
            sorted_labels,
            label_to_index,
            index_to_label,
        )

    def split_data(
        self, data: DataFrame, test_size_from_full: float = 0.2, val_size_from_remainder: float = 0.25
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        train_data: DataFrame
        test_data: DataFrame
        val_data: DataFrame
        train_data, test_data = train_test_split(data, test_size=test_size_from_full, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=val_size_from_remainder, random_state=42)

        return train_data, val_data, test_data  # type: ignore
