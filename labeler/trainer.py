import logging
from .input_processor import InputProcessor

from transformers import Trainer, TrainingArguments, PreTrainedModel, AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
from datasets import Dataset  # type: ignore
from pathlib import Path

from torch import backends


class TrainingConfig:
    def __init__(
        self,
        output_directory: Path,
        labels_csv: Path,
        model_type: str = "bert-base-uncased",
        batch_size: int = 2,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
    ):
        self.output_directory = output_directory
        self.labels_csv = labels_csv
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs


class ClassifierTrainer:
    def __init__(self, log_files_directory: Path, training_config: TrainingConfig):
        self.log_files_directory = log_files_directory
        self.training_config = training_config
        self.input_processor = InputProcessor(log_files_directory, training_config.labels_csv)
        self.tokenizer = AutoTokenizer.from_pretrained(training_config.model_type)

        self.dataframe, self.labels, self.label_to_index, self.index_to_label = self.input_processor.read_data(
            self.tokenizer
        )
        self.train_dataframe, self.val_dataframe, self.test_dataframe = self.input_processor.split_data(self.dataframe)
        self.train_data = Dataset.from_pandas(self.train_dataframe)
        self.val_data = Dataset.from_pandas(self.val_dataframe)
        self.test_data = Dataset.from_pandas(self.test_dataframe)

        self.num_labels = len(self.labels)

        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            self.training_config.model_type,
            problem_type="multi_label_classification",
            num_labels=self.num_labels,
            id2label=self.index_to_label,
            label2id=self.label_to_index,
        )

    def train_v2(self) -> None:
        training_args = TrainingArguments(
            output_dir=str(self.training_config.output_directory),
            logging_dir="./logs",
            logging_steps=1,
            evaluation_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            num_train_epochs=self.training_config.num_epochs,
            warmup_steps=500,
            weight_decay=0.01,
            use_mps_device=backends.mps.is_available(),  # type: ignore
            label_names=self.labels,
            load_best_model_at_end=True,
            report_to=["tensorboard"],
            run_name="labeler",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,  # type: ignore
            eval_dataset=self.val_data,  # type: ignore
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.evaluate()

        predictions = trainer.predict(test_dataset=self.test_data)  # type: ignore
        logging.info(predictions.predictions)  # types: ignore

    def save_pretrained(self, save_directory: Path):
        self.model.save_pretrained(save_directory, state_dict=self.model.state_dict())
        self.tokenizer.save_pretrained(save_directory)
        logging.info("Model saved to '%s'", save_directory)
