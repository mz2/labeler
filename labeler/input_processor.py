import concurrent.futures
import logging

from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from pandas import DataFrame, Series, read_csv
from sklearn.model_selection import train_test_split
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore


class InputProcessor:
    def __init__(
        self,
        log_files_dir: Path,
        labels_file_path: Path,
        min_samples_per_label: int,
        delimiter: str = ";",
        label_delimiter: str = " ",
    ):
        self.log_files_dir: Path = log_files_dir
        self.labels_file_path: Path = labels_file_path
        self.csv_delimiter = delimiter
        self.label_delimiter = label_delimiter
        self.min_samples_per_label = min_samples_per_label

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

    @staticmethod
    def filter_labels_by_count(df: DataFrame, min_count: int) -> DataFrame:
        label_counts = df["labels"].apply(Series).stack().value_counts()
        included_labels: List[str] = label_counts[label_counts >= min_count].index.tolist()
        included_labels_set = set(included_labels)
        filtered_df = df[df["labels"].apply(lambda x: len(set(x).intersection(included_labels_set)) > 0)]
        return filtered_df

    def read_data(
        self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    ) -> Tuple[DataFrame, List[str], Dict[str, int], Dict[int, str]]:
        labels_data = read_csv(
            self.labels_file_path, delimiter=self.csv_delimiter, dtype={"path": "string", "labels": "string"}
        )
        labels_data["labels"] = labels_data["labels"].str.split().sort_values()

        labels_data = InputProcessor.filter_labels_by_count(labels_data, min_count=self.min_samples_per_label)

        sorted_labels = self.sorted_labels(labels_data)
        label_to_index = {label: i for i, label in enumerate(sorted_labels)}
        index_to_label = {i: label for i, label in enumerate(sorted_labels)}

        # labels_data["label_ids"] = labels_data["labels"].apply(lambda x: [label_to_index[label] for label in x])

        texts: List[str] = []
        labels: List[List[float]] = []
        input_ids = []
        token_type_ids = []
        attention_masks = []

        def tokenize_text(row: Series):  # type: ignore
            path: str = row["path"]
            with open(self.log_files_dir.joinpath(path), "r") as f:
                content = f.read()
                content = self._preprocess_log(content)
                texts.append(content)
                label_list = InputProcessor.convert_labels_to_binary(
                    row["labels"], sorted_labels=sorted_labels, label_to_index=label_to_index
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

        logging.info("Tokenizing input…")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(tokenize_text, row) for _, row in labels_data.iterrows()]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()

        return (
            DataFrame(
                {
                    "text": texts,
                    "path": labels_data["path"],
                    "labels": labels,
                    # "label_ids": labels_data["label_ids"],
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "attention_masks": attention_masks,
                }
            ),
            sorted_labels,
            label_to_index,
            index_to_label,
        )

    def _read_file(self, path: str) -> str:
        with open(self.log_files_dir.joinpath(path), "r") as f:
            content = f.read()
        return content

    @staticmethod
    def _tokenize_input(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        text: str,
        truncation: bool = True,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> BatchEncoding:
        return tokenizer.encode_plus(
            text, truncation=truncation, padding=padding, add_special_tokens=add_special_tokens, return_tensors="pt"
        )

    @staticmethod
    def sorted_labels(labels_data: DataFrame) -> List[str]:
        labels = set()
        for _, row in labels_data.iterrows():
            labels.update(row["labels"])
        return sorted(labels)

    def split_data(
        self, data: DataFrame, test_size_from_full: float = 0.2, val_size_from_remainder: float = 0.25
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        train_data, test_data = train_test_split(data, test_size=test_size_from_full, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=val_size_from_remainder, random_state=42)

        return train_data, val_data, test_data  # type: ignore
