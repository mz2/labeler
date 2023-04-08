import argparse
import logging
from pathlib import Path

from labeler.model_trainer import ClassifierTrainer, TrainingConfig
from labeler.model_evaluator import ClassifierEvaluator

logging.basicConfig(level=logging.INFO)

def main(args: argparse.Namespace):
    if args.mode == "train": 
        labels_path = Path(args.labels) # type: ignore
        trainer = ClassifierTrainer(Path(args.logs), TrainingConfig(Path(args.labels), batch_size=16, learning_rate=2e-5, num_epochs=10))
        trainer.train()
        trainer.save_pretrained(Path(args.save))

    elif args.mode == "infer":
        evaluator = ClassifierEvaluator(Path(args.load))
        probabilities = evaluator.infer(args.log)
        print(probabilities)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilabel BERT classifier for log files")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation: train or infer")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--save", help="Directory to save the fine-tuned model and tokenizer", required=True)
    train_parser.add_argument("--logs", help="Path to a directory with log files for training", required=True)
    train_parser.add_argument("--labels", help="Path to a CSV file with labels of input files", required=True)

    infer_parser = subparsers.add_parser("infer", help="Run inference on log files")
    infer_parser.add_argument("--load", help="Directory to load the fine-tuned model and tokenizer from", required=True)
    infer_parser.add_argument("--logs", help="Path to a directory with log files for inference", required=True)

    args = parser.parse_args()
    main(args)
    