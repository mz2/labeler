#!/usr/bin/env python3

import argparse
import sys
from typing import List
from labeler.parser.processor import train, matches, filter_uninteresting_lines
from labeler.parser.miner import create_template_miner
from labeler.tokenizer import tokenized_text


def preprocess_file(input_file: str, window_size: int) -> List[str]:
    with open(input_file, "r") as f:
        text = f.readlines()

    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Log file parser using Drain3")
    parser.add_argument("input_files", nargs="+", type=str, help="Paths to the log files, use '-' for stdin")
    parser.add_argument("-s", "--size", type=int, default=512, help="Size of chunks to tokenize (default: 512).")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer model to use (default: bert-base-uncased).",
    )
    parser.add_argument("-b", "--show_boundaries", action="store_true", help="Show chunk boundaries in output.")
    parser.add_argument(
        "-p", "--preprocess_only", action="store_false", help="Output only preprocessed text instead of tokenized text."
    )
    parser.add_argument(
        "-n",
        "--window_size",
        type=int,
        default=10,
        help="""Size of window for filtering around potentially interesting log lines
                (right now treated as lines that do not look like DEBUG log lines).
                If negative value passed in, no filtering should be done.""",
    )

    args = parser.parse_args()

    template_miner = create_template_miner()

    for input_file in args.input_files:
        if input_file == "-":
            file_context = sys.stdin
        else:
            file_context = open(input_file)

        with file_context as f:
            lines = [line.strip() for line in f]

            if args.window_size >= 0:
                lines = filter_uninteresting_lines(lines, args.window_size)

            train(template_miner, lines)

            for match in matches(lines, template_miner):
                if args.preprocess_only:
                    print(match)
                else:
                    tokenized = tokenized_text(match, args.size, args.model, args.show_boundaries)
                    print(tokenized)


if __name__ == "__main__":
    main()
