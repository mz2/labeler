import os
import csv
import shutil
import argparse
import tarfile
import tempfile
from typing import List


def get_directories(input_dir: str) -> List[str]:
    return [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]


def get_files(directory: str, suffix: str = "_0.txt") -> List[str]:
    return [f for f in os.listdir(directory) if f.endswith(suffix)]


def main(input_path: str, output_dir: str, csv_filename: str):
    os.makedirs(output_dir, exist_ok=True)

    if input_path.endswith(".tar.gz"):
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(input_path, "r:gz") as tar:
                tar.extractall(temp_dir)
            process_files(os.path.join(temp_dir, "data"), output_dir, csv_filename)
    else:
        process_files(input_path, output_dir, csv_filename)


def process_files(input_dir: str, output_dir: str, csv_filename: str):
    directories = get_directories(input_dir)
    csv_rows = [["path", "labels"]]

    for directory in directories:
        bug_id = os.path.basename(directory)
        output_subdir = os.path.join(output_dir, bug_id)

        os.makedirs(output_subdir, exist_ok=True)

        for file in get_files(directory):
            src = os.path.join(directory, file)
            dst = os.path.join(output_subdir, file)

            shutil.copy(src, dst)
            csv_rows.append([os.path.join(bug_id, file), bug_id])

    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=";")
        for row in csv_rows:
            csv_writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input directory or tar.gz file")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    parser.add_argument("-c", "--csv", required=True, help="Filename for the output CSV file")

    args = parser.parse_args()
    main(args.input, args.output, args.csv)
