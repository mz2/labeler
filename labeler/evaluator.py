import csv
import json
import sys
from typing import IO, List, Optional, Tuple
from .device import get_device

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
from pathlib import Path


class ClassifierEvaluator:
    def __init__(self, directory: Path) -> None:
        self.tokenizer: AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(directory)  # type: ignore

        self.model: AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory, problem_type="multi_label_classification"
        )  # type: ignore

    def __read_log_files(self, log_file_path: Path | List[str]) -> Tuple[List[Path] | None, List[str]]:
        if isinstance(log_file_path, Path):
            if log_file_path.is_dir():
                log_files = [Path(log_file) for log_file in log_file_path.glob("**/*") if log_file.is_file()]
            elif log_file_path.is_file():
                log_files = [Path(log_file_path)]
            else:
                raise ValueError("Invalid log file path (should be a file or a directory)")

            log_texts: list[str] = []

            for log_file in log_files:
                with open(log_file, "r") as f:
                    log_text = f.read()
                log_texts.append(log_text)

            return log_files, log_texts
        else:
            log_texts = log_file_path
            return None, log_texts

    def infer(
        self, log_file_path: Path | List[str], threshold: float = 0.5, filter_empty: bool = False
    ) -> List[Tuple[str, str, List[float]]]:
        self.model.eval()  # type: ignore
        device = get_device()
        self.model.to(device)  # type: ignore

        file_paths, log_texts = self.__read_log_files(log_file_path)

        results = []
        for file_path, text in zip(file_paths or [None] * len(log_texts), log_texts):
            probabilities = (
                torch.sigmoid(
                    self.model(**self.tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)).logits  # type: ignore
                )
                .detach()
                .cpu()
                .numpy()[0]
                .tolist()
            )

            # Get labels above the threshold probability
            labels_above_threshold = [
                self.model.config.id2label[label_id]  # type: ignore
                for label_id, probability in enumerate(probabilities)
                if probability >= threshold
            ]
            label_names = " ".join(labels_above_threshold)

            if filter_empty and not labels_above_threshold:
                continue

            results.append((str(file_path), label_names, probabilities))

        return results

    def write_to_json(self, results: List[Tuple[str, str, List[float]]], output_file: Optional[Path] = None) -> None:
        json_data = [
            {
                "path": file_path,
                "labels": label_names,
                "probabilities": {self.model.config.id2label[i]: prob for i, prob in enumerate(probabilities)},  # type: ignore
            }
            for file_path, label_names, probabilities in results
        ]

        if output_file:
            with open(output_file, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
        else:
            print(json.dumps(json_data, indent=4))

    def write_to_csv(
        self, results: List[Tuple[str, str, List[float]]], output_file: Optional[Path] = None, delimiter: str = ","
    ) -> None:
        if output_file:
            with open(output_file, "w", newline="") as csvfile:
                self._write_csv_data(results, csvfile, delimiter)
        else:
            self._write_csv_data(results, sys.stdout, delimiter)

    def _write_csv_data(
        self, results: List[Tuple[str, str, List[float]]], write_handle: IO[str], delimiter: str
    ) -> None:
        csv_writer = csv.writer(write_handle, delimiter=delimiter)

        header = ["Path", "Labels"]
        for _, label_name in self.model.config.id2label.items():  # type: ignore
            header.append(label_name)
        csv_writer.writerow(header)

        for result in results:
            file_path, label_names, probabilities = result
            csv_writer.writerow([file_path, label_names, *probabilities])
