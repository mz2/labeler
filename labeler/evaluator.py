from typing import List
from .device import get_device

import torch

from transformers import BertForSequenceClassification, BertTokenizer # type: ignore
from transformers import BertTokenizer, BertForSequenceClassification # type: ignore
from pathlib import Path

class ClassifierEvaluator:

    def __init__(self, directory: Path) -> None:
        self.tokenizer: BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(directory) # type: ignore

        self.model: BertForSequenceClassification
        self.model = BertForSequenceClassification.from_pretrained(directory) # type: ignore

    def __read_log_files(self, log_file_path: Path | List[str]):
        if isinstance(log_file_path, Path):
            if log_file_path.is_dir():
                log_files = [str(log_file) for log_file in log_file_path.glob('*')]
            elif log_file_path.is_file():
                log_files = [str(log_file_path)]
            else:
                raise ValueError('Invalid log file path (should be a file or a directory)')
            
            log_texts: list[str] = []

            for log_file in log_files:
                with open(log_file, 'r') as f:
                    log_text = f.read()
                log_texts.append(log_text)

        else:
            log_texts = log_file_path
        
        return log_texts

    def infer(self, log_file_path: Path | List[str]) -> List[List[float]]:
        self.model.eval()
        device = get_device()
        self.model.to(device) # type: ignore

        log_texts = self.__read_log_files(log_file_path)

        return [
            torch.sigmoid(self.model(**self.tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device))
                        .logits).detach().cpu().numpy()[0].tolist()
            for text in log_texts
        ]
