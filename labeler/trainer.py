import logging
import torch

from .input_processor import InputProcessor

from transformers import Trainer, TrainingArguments, PreTrainedModel, AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
from datasets import Dataset  # type: ignore
from pathlib import Path
from datetime import datetime


class TrainingConfig:
    def __init__(
        self,
        output_directory: Path,
        labels_csv: Path,
        model_type: str,
        batch_size: int = 2,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        min_samples_per_label: int = 1,
    ):
        self.output_directory = output_directory
        self.labels_csv = labels_csv
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.min_samples_per_label = min_samples_per_label


class ClassifierTrainer:
    def __init__(self, log_files_directory: Path, training_config: TrainingConfig):
        self.log_files_directory = log_files_directory
        self.training_config = training_config
        self.input_processor = InputProcessor(
            log_files_directory, training_config.labels_csv, min_samples_per_label=training_config.min_samples_per_label
        )
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

    # def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
    #     logging.info("Computing metrics...")
    #     logging.info(eval_pred)
    #     logits, labels = eval_pred
    #     predictions = logits.argmax(axis=-1)  # type: ignore
    #     tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    #     accuracy = (tp + tn) / (tp + tn + fp + fn)
    #     fpr = fp / (fp + tn)
    #     fnr = fn / (fn + tp)
    #     probabilities = np.exp(logits)[:, 1] / np.exp(logits).sum(axis=-1)
    #     return {"accuracy": accuracy, "fpr": fpr, "fnr": fnr, "probabilities": probabilities}

    # def roc_auc_curve(self, eval_output: PredictionOutput):
    #     logging.info(eval_output)

    #     fpr, tpr, _ = roc_curve(eval_output.label_ids, eval_output.probabilities)  # type: ignore
    #     roc_auc = auc(fpr, tpr)

    #     plt.plot(fpr, tpr, color="darkorange", label="ROC curve (area = %0.2f)" % roc_auc)
    #     plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title("Receiver operating characteristic (ROC) curve")
    #     plt.legend(loc="lower right")

    #     output_dir = self.training_config.output_directory
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)

    #     output_path = os.path.join(output_dir, "roc_curve.pdf")
    #     plt.savefig(output_path)

    def train(self) -> None:
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"labeler_{timestamp_str}"

        training_args = TrainingArguments(
            output_dir=str(self.training_config.output_directory),
            logging_dir="./logs",
            logging_steps=1,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            num_train_epochs=self.training_config.num_epochs,
            warmup_steps=500,
            weight_decay=0.01,
            use_mps_device=torch.backends.mps.is_available(),  # type: ignore
            label_names=self.labels,
            load_best_model_at_end=True,
            report_to=["tensorboard"],
            run_name=run_name,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,  # type: ignore
            eval_dataset=self.val_data,  # type: ignore
            tokenizer=self.tokenizer,
            # compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.evaluate()

        logging.info("ROC AUC curve coming up...")
        logging.info(self.test_data)
        predictions = trainer.predict(test_dataset=self.test_data)  # type: ignore
        # self.roc_auc_curve(predictions)

        logging.info(predictions.predictions)  # types: ignore

    def save_pretrained(self, save_directory: Path):
        self.model.save_pretrained(save_directory, state_dict=self.model.state_dict())
        self.tokenizer.save_pretrained(save_directory)
        logging.info("Model saved to '%s'", save_directory)
