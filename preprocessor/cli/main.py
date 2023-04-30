#!/usr/bin/env python3

import argparse
from labeler.preprocessor import preprocessed_text
from labeler.tokenizer import tokenized_text


def preprocess_file(input_file: str, window_size: int) -> str:
    with open(input_file, "r") as f:
        text = f.read()

    return preprocessed_text(text, window_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and tokenize a text file.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input text file.")
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
        "-p", "--preprocess_only", action="store_true", help="Output only preprocessed text instead of tokenized text."
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

    preprocessed_text = preprocess_file(args.input, args.window_size)

    if args.preprocess_only:
        print(preprocessed_text)
    else:
        tokenized = tokenized_text(preprocessed_text, args.size, args.model, args.show_boundaries)
        print(tokenized)


if __name__ == "__main__":
    main()
