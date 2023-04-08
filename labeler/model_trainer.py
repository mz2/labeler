from typing import Any, List
from .log_processor import LogProcessor
from .device import get_device

import torch
import numpy as np

from transformers import BertForSequenceClassification, BertTokenizer # type: ignore
from transformers import BertTokenizer, BertForSequenceClassification # type: ignore
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

class TrainingConfig:
    def __init__(self, labels: Path, batch_size: int = 16, learning_rate: float = 2e-5, num_epochs: int = 10):
        self.labels = labels
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

class ClassifierTrainer:
    def __init__(self, log_files_directory: Path, training_config: TrainingConfig):
        self.log_files_directory = log_files_directory
        self.training_config = training_config
        self.log_processor = LogProcessor(log_files_directory, training_config.labels)

        self.data = self.log_processor.read_logs()
        train_data, _, _ = self.log_processor.split_data(self.data)
        self.train_texts, self.train_bug_labels = self.log_processor.extract_texts_and_labels(train_data)

        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # type: ignore
        self.num_labels = len(set(self.train_bug_labels))

        self.model: BertForSequenceClassification
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels) # type: ignore
        
    def __convert_labels_to_binary(
        self, bug_labels: List[str], num_labels: int
    ) -> np.ndarray[float, Any]:
        bug_labels_list = [list(map(int, labels.split(','))) for labels in bug_labels]
        binary_labels = np.zeros((len(bug_labels), num_labels), dtype=int)

        for i, labels in enumerate(bug_labels_list):
            for label in labels:
                binary_labels[i, label] = 1

        return binary_labels

    def train(self) -> None:
        train_encodings = tokenizer(train_texts, truncation=True, padding=True) # type: ignore
        train_labels = self.__convert_labels_to_binary(self.train_bug_labels, self.num_labels)

        train_dataset = TensorDataset(torch.tensor(train_encodings["input_ids"]),
                                      torch.tensor(train_encodings["attention_mask"]),
                                      torch.tensor(train_labels))

        train_loader = DataLoader(train_dataset, batch_size=self.training_config.batch_size, shuffle=True) # type: ignore

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config.learning_rate)

        self.model.train()
        device = get_device()
        self.model.to(device) # type: ignore

        for _ in range(self.training_config.num_epochs):
            for batch in train_loader:
                input_ids, attention_mask, labels = [t.to(device) for t in batch]
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward() 
                optimizer.step()

    def save_pretrained(self, save_directory: Path):
        self.model.save_pretrained(save_directory) # type: ignore
        self.tokenizer.save_pretrained(save_directory) # type: ignore