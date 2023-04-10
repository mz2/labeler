import argparse
import logging
from pathlib import Path
from enum import Enum

from labeler.trainer import ClassifierTrainer, TrainingConfig
from labeler.evaluator import ClassifierEvaluator

logging.basicConfig(level=logging.INFO)


class OutputFormat(Enum):
    UNKNOWN = "unknown"
    CSV = "csv"
    JSON = "json"

    @classmethod
    def infer_from_output(cls, output: Path):
        if output.suffix == ".csv":
            return cls.CSV
        elif output.suffix == ".json":
            return cls.JSON
        else:
            raise ValueError("Unsupported output file extension. Use '.csv' or '.json'.")


def main(args: argparse.Namespace):
    if args.mode == "train":
        trainer = ClassifierTrainer(
            Path(args.logs),
            TrainingConfig(
                output_directory=Path(args.save),
                labels_csv=Path(args.labels),
                model_type=args.model_type,
                min_samples_per_label=args.min_samples,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
            ),
        )
        trainer.train()
        trainer.save_pretrained(Path(args.save))

    elif args.mode == "infer":
        evaluator = ClassifierEvaluator(Path(args.load))
        output = evaluator.infer(Path(args.logs), threshold=args.threshold, filter_empty=args.filter)
        output_file = None if args.output == "-" else args.output

        # output to CSV by default since it's assumed easier to read from stdout
        output_format = OutputFormat.CSV

        if args.format == OutputFormat.UNKNOWN and output_file is not None:
            output_format = OutputFormat.infer_from_output(Path(args.output))
        elif args.format != OutputFormat.UNKNOWN:
            output_format = args.format

        if output_format == OutputFormat.JSON:
            evaluator.write_to_json(output, output_file)
        elif output_format == OutputFormat.CSV:
            evaluator.write_to_csv(output, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilabel BERT classifier for log files")
    parser.add_argument("--model-type", help="Transformer model to use", default="bert-base-uncased")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation: train or infer")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--save", help="Directory to save the fine-tuned model and tokenizer", required=True)
    train_parser.add_argument("--logs", help="Path to a directory with log files for training", required=True)
    train_parser.add_argument("--labels", help="Path to a CSV file with labels of input files", required=True)
    train_parser.add_argument("--epochs", help="Path to a CSV file with labels of input files", default=10, type=int)
    train_parser.add_argument("--min-samples", help="Minimum number of samples", default=1, type=int)

    train_parser.add_argument("--batch-size", help="Batch size", default=2, type=int)

    infer_parser = subparsers.add_parser("infer", help="Run inference on log files")
    infer_parser.add_argument("--load", help="Directory to load the fine-tuned model and tokenizer from", required=True)
    infer_parser.add_argument("--logs", help="Path to a directory with log files for inference", required=True)
    infer_parser.add_argument("--threshold", help="Probability threshold", default=0.5, type=float)
    infer_parser.add_argument("--output", help="File to save the output CSV (use '-' for stdout)", default="-")
    infer_parser.add_argument(
        "--filter", help="Filter out results where no prediction is made", type=bool, default=False
    )
    infer_parser.add_argument(
        "--format",
        help="Output format (csv or json)",
        required=True,
        type=OutputFormat,
        choices=list(OutputFormat),
        default=OutputFormat.UNKNOWN,
        metavar="FORMAT",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    main(args)
