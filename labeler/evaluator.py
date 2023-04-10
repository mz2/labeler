import csv
import json
import logging
import sys
from typing import IO, List, Optional, Tuple

import torch

from transformers import PreTrainedModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore
from pathlib import Path


class ClassifierEvaluator:
    def __init__(
        self,
        directory: Path,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    ) -> None:
        if tokenizer is None:
            logging.debug("Loading tokenizer from %s", directory)
            self.tokenizer = AutoTokenizer.from_pretrained(directory)  # type: ignore
        else:
            self.tokenizer = tokenizer

        if model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                directory, problem_type="multi_label_classification"
            )
        else:
            self.model = model

    def read_files(self, file_path: Path | List[str]) -> Tuple[List[Path] | None, List[str]]:
        if isinstance(file_path, Path):
            if file_path.is_dir():
                files = [Path(log_file) for log_file in file_path.glob("**/*") if log_file.is_file()]
            elif file_path.is_file():
                files = [Path(file_path)]
            else:
                raise ValueError("Invalid log file path (should be a file or a directory)")

            texts: list[str] = []

            for log_file in files:
                with open(log_file, "r") as f:
                    log_text = f.read()
                texts.append(log_text)

            return files, texts
        else:
            log_texts = file_path
            return None, log_texts

    def predict_labels(self, texts: List[str], threshold: float = 0.5) -> Tuple[List[List[str]], torch.Tensor]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        binary_preds = probs > threshold
        predicted_labels = []
        for preds in binary_preds:
            label_indices = torch.where(preds)[0]
            labels = [self.model.config.id2label[label_idx.item()] for label_idx in label_indices]  # type: ignore
            predicted_labels.append(labels)
        return (predicted_labels, probs)

    def infer(
        self, log_file_path: Path | List[str], threshold: float = 0.5, filter_empty: bool = False
    ) -> List[Tuple[str, str, List[float]]]:
        file_paths, texts = self.read_files(log_file_path)

        predicted_labels, logits = self.predict_labels(texts, threshold=threshold)

        results = []
        for file_path, labels, probs in zip(file_paths or [None] * len(texts), predicted_labels, logits):
            label_names = " ".join(labels)
            if filter_empty and not labels:
                continue
            results.append((str(file_path), label_names, probs.tolist()))

        return results

    def write_to_json(self, results: List[Tuple[str, str, List[float]]], output_file: Optional[Path] = None) -> None:
        json_data = [
            {
                "path": file_path,
                "labels": label_names,
                "probabilities": {
                    self.model.config.id2label[i]: prob  # type: ignore
                    for i, prob in enumerate(probabilities)  # type: ignore
                },
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
