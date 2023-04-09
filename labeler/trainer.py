import logging
from typing import Any, Dict, List, Tuple
from .input_processor import InputProcessor
from .device import get_device

import torch
import numpy as np

from transformers import BertForSequenceClassification  # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


class Metrics:
    def __init__(self, average_loss: float, fpr: float, fnr: float, accuracy: float):
        self.average_loss = average_loss
        self.fpr = fpr
        self.fnr = fnr
        self.accuracy = accuracy

    def __str__(self):
        return f"Average loss: {self.average_loss}, FPR: {self.fpr}, FNR: {self.fnr}, Accuracy: {self.accuracy}"


class TrainingConfig:
    def __init__(
        self,
        labels: Path,
        model_type: str = "bert-base-uncased",
        batch_size: int = 2,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
    ):
        self.labels = labels
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs


class ClassifierTrainer:
    def __init__(self, log_files_directory: Path, training_config: TrainingConfig):
        self.log_files_directory = log_files_directory
        self.training_config = training_config
        self.input_processor = InputProcessor(log_files_directory, training_config.labels)

        self.data = self.input_processor.read_logs()
        train_data, val_data, test_data = self.input_processor.split_data(self.data)

        self.train_texts, self.train_labels = self.input_processor.extract_texts_and_labels(train_data)
        self.val_texts, self.val_labels = self.input_processor.extract_texts_and_labels(val_data)
        self.test_texts, self.test_labels = self.input_processor.extract_texts_and_labels(test_data)

        self.tokenizer = AutoTokenizer.from_pretrained(training_config.model_type)  # type: ignore
        self.num_labels = len(set(self.train_labels))
        self.label_to_index, self.index_to_label = self.__create_label_to_index_map(self.train_labels)

        # Discard data to ensure that the lengths of the data sets are divisible by the batch size
        for data_set, data_set_name in [
            (self.train_texts, "training"),
            (self.val_texts, "validation"),
            (self.test_texts, "test"),
        ]:
            remainder = len(data_set) % self.training_config.batch_size
            if remainder != 0:
                print(f"Discarding {remainder} samples from the {data_set_name} data set.")
                data_set_size = len(data_set) - remainder
                if data_set_name == "training":
                    self.train_texts = self.train_texts[:data_set_size]
                    self.train_labels = self.train_labels[:data_set_size]
                elif data_set_name == "validation":
                    self.val_texts = self.val_texts[:data_set_size]
                    self.val_labels = self.val_labels[:data_set_size]
                elif data_set_name == "test":
                    self.test_texts = self.test_texts[:data_set_size]
                    self.test_labels = self.test_labels[:data_set_size]

        self.model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
            self.training_config.model_type,
            problem_type="multi_label_classification",
            num_labels=self.num_labels,
            id2label=self.index_to_label,
            label2id=self.label_to_index,
        )

    def __create_label_to_index_map(self, labels_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        unique_labels: set[str] = {label for labels in labels_list for label in labels.split(" ")}
        sorted_labels = sorted(unique_labels)
        return (
            {label: i for i, label in enumerate(sorted_labels)},
            {i: label for i, label in enumerate(sorted_labels)},
        )

    def __convert_labels_to_binary(self, labels_list: List[str], num_labels: int) -> np.ndarray[float, Any]:
        binary_labels = np.zeros((len(labels_list), num_labels), dtype=np.float32)

        for i, labels in enumerate(labels_list):
            for label in labels.split(" "):
                binary_labels[i, self.label_to_index[label]] = 1.0

        return binary_labels

    def evaluate(self, dataloader: DataLoader[Any]) -> Metrics:
        device = get_device()
        self.model.eval()  # type: ignore
        total_loss: float = 0
        total_steps: float = 0
        total_tp: float = 0
        total_fp: float = 0
        total_tn: float = 0
        total_fn: float = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = [t.to(device) for t in batch]

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)  # type: ignore
                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1

                logits = outputs.logits
                preds = torch.sigmoid(logits) > 0.5
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()

                tp = np.sum((labels == 1) & (preds == 1))
                fp = np.sum((labels == 0) & (preds == 1))
                tn = np.sum((labels == 0) & (preds == 0))
                fn = np.sum((labels == 1) & (preds == 0))
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn

        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)

        return Metrics(total_loss / total_steps, 1 - precision, 1 - recall, accuracy)

    def train(self) -> Metrics:
        train_encodings = self.tokenizer(self.train_texts, truncation=True, padding=True)
        train_labels = self.__convert_labels_to_binary(self.train_labels, self.num_labels)
        train_dataset = TensorDataset(
            torch.tensor(train_encodings["input_ids"]),
            torch.tensor(train_encodings["attention_mask"]),
            torch.tensor(train_labels),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.training_config.batch_size, shuffle=True)

        val_encodings = self.tokenizer(self.val_texts, truncation=True, padding=True)
        val_labels = self.__convert_labels_to_binary(self.val_labels, self.num_labels)
        val_dataset = TensorDataset(
            torch.tensor(val_encodings["input_ids"]),
            torch.tensor(val_encodings["attention_mask"]),
            torch.tensor(val_labels),
        )
        val_loader = DataLoader(val_dataset, batch_size=self.training_config.batch_size, shuffle=False)

        test_encodings = self.tokenizer(self.test_texts, truncation=True, padding=True)
        test_labels = self.__convert_labels_to_binary(self.test_labels, self.num_labels)
        test_dataset = TensorDataset(
            torch.tensor(test_encodings["input_ids"]),
            torch.tensor(test_encodings["attention_mask"]),
            torch.tensor(test_labels),
        )
        test_loader = DataLoader(test_dataset, batch_size=self.training_config.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config.learning_rate)  # type: ignore

        self.model.train()  # type: ignore
        device = get_device()
        self.model.to(device)  # type: ignore

        for epoch in range(self.training_config.num_epochs):
            for batch in train_loader:
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)  # type: ignore
                logging.debug(outputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            val_output = self.evaluate(val_loader)
            logging.info(f"Validation loss after epoch {epoch + 1}: {val_output}")

        test_output = self.evaluate(test_loader)
        logging.info(f"Test loss after training: {test_output}")

        return test_output

    def save_pretrained(self, save_directory: Path):
        self.model.save_pretrained(save_directory)  # type: ignore
        self.tokenizer.save_pretrained(save_directory)  # type: ignore
        logging.info("Model saved to '%s'", save_directory)
