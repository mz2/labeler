#!/usr/bin/env python3

import argparse
import sys
from parser.processor import train, matches
from parser.miner import create_template_miner


def main() -> None:
    parser = argparse.ArgumentParser(description="Log file parser using Drain3")
    parser.add_argument("input_files", nargs="+", type=str, help="Paths to the log files, use '-' for stdin")
    args = parser.parse_args()

    template_miner = create_template_miner()

    for input_file in args.input_files:
        if input_file == "-":
            file_context = sys.stdin
        else:
            file_context = open(input_file)

        with file_context as f:
            lines = [line.strip() for line in f]
            train(template_miner, lines)

            for match in matches(lines, template_miner):
                print(match)


if __name__ == "__main__":
    main()
