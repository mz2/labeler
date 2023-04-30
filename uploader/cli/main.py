#!/usr/bin/env python3
import argparse
import json
import requests
from typing import Any, List

from labeler.batch_data import batch_data


def upload_to_label_studio(input_files: List[str], auth: str, host: str, proj: int, size: int, overlap: int) -> None:
    if not host.startswith(("http://", "https://")):
        host = f"https://{host}"

    url = f"{host}/api/projects/{proj}/import"
    headers = {"Authorization": f"Token {auth}", "Content-Type": "application/json"}

    for input_file in input_files:
        with open(input_file, "r") as f:
            text = f.read()
            batches = batch_data(text, size, overlap)

            # Print the number of batches
            print(f"Uploading {len(batches)} batches from file '{input_file}'...")

            for batch in batches:
                data = [{"text": batch}]
                response = requests.post(url, headers=headers, data=json.dumps(data))

                if response.status_code == 201:
                    print(f"Batch from file '{input_file}' uploaded successfully.")
                else:
                    print(f"Error uploading batch from file '{input_file}'. Status code: {response.status_code}")
                    print("Response:", response.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload files to Label Studio instance.")
    parser.add_argument("input_files", nargs="+", help="The input files to be uploaded.")
    parser.add_argument("-a", "--auth", required=True, help="The authorization token for Label Studio.")
    parser.add_argument("-lh", "--host", required=True, help="The Label Studio hostname.")
    parser.add_argument("-p", "--proj", type=int, required=True, help="The project ID in Label Studio.")
    parser.add_argument("-s", "--size", type=int, default=512, help="Max size per batch in number of lines.")
    parser.add_argument("-o", "--overlap", type=int, default=1, help="The number of overlapping lines between batches.")
    parser.add_argument(
        "-b",
        "--bytes",
        type=int,
        default=2048,
        help="Max size per batch in # bytes (only approximately enforced when overlap is specified).",
    )
    args: Any = parser.parse_args()

    upload_to_label_studio(args.input_files, args.auth, args.host, args.proj, args.size, args.overlap)


if __name__ == "__main__":
    main()
