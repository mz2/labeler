import argparse
import os
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

from .log_processor import LogProcessor
from .device import get_device
import numpy as np

class MultilabelBertLogClassifier:
    def __init__(self, log_files_directory: str, log_files_labels: str, batch_size: int = 16, learning_rate: float = 2e-5, num_epochs: int = 10):
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.num_epochs: int = num_epochs
        self.log_processor: LogProcessor = LogProcessor(log_files_directory, log_files_labels)
        self.model: BertForSequenceClassification = None
        self.tokenizer: BertTokenizer = None

    def __convert_labels_to_binary(
        self, bug_labels: pd.Series, num_labels: int
    ) -> np.ndarray:
        bug_labels_list = [list(map(int, labels.split(','))) for labels in bug_labels]
        binary_labels = np.zeros((len(bug_labels), num_labels), dtype=int)

        for i, labels in enumerate(bug_labels_list):
            for label in labels:
                binary_labels[i, label] = 1

        return binary_labels

    def train(self) -> None:
        data = self.log_processor.read_logs()
        train_data, val_data, test_data = self.log_processor.split_data(data)
        train_texts, train_bug_labels = self.log_processor.extract_texts_and_labels(train_data)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        num_labels = len(set(train_bug_labels))
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        train_labels = self.__convert_labels_to_binary(train_bug_labels, num_labels)

        train_dataset = TensorDataset(torch.tensor(train_encodings["input_ids"]),
                                      torch.tensor(train_encodings["attention_mask"]),
                                      torch.tensor(train_labels))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()

        device = get_device()
   
        self.model.to(self.device)

        for epoch in range(self.num_epochs):
            for batch in train_loader:
                input_ids, attention_mask, labels = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def infer(self, log_file: str):
        self.model.eval()

        device = get_device()

        self.model.to(device)

        with open(log_file, 'r') as f:
            log_text = f.read()

        encoding = self.tokenizer(log_text, truncation=True, padding=True, return_tensors='pt')
        input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
        probabilities = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]
        return probabilities

def main(args):
    classifier = MultilabelBertLogClassifier()

    if args.mode == "train":
        classifier.train()
        classifier.save_pretrained(args.save)

    elif args.mode == "infer":
        classifier.load_pretrained(args.load)
        probabilities = classifier.infer(args.log)
        print(f"Probabilities: {probabilities}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilabel BERT Classifier for Log Files")
    parser.add_argument("mode", choices=["train", "infer"], help="Mode of operation: train or infer")
    parser.add_argument("--save", help="Directory to save the fine-tuned model and tokenizer")
    parser.add_argument("--load", help="Directory to load the fine-tuned model and tokenizer from")
    parser.add_argument("--log", help="Path to the unlabelled log file for inference")

    args = parser.parse_args()
    main(args)
