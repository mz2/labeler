# labeler - ML driven labeler of files ![Build status](https://github.com/mz2/labeler/actions/workflows/test.yml/badge.svg)

A transformer model fine-tuning based tool for associating textual data with one or more labels (multi-label classifier of text data).

Tested with [`bert-base-uncased`](https://huggingface.co/bert-base-uncased), but should work beside BERT with a lot of other transformers present in the huggingface [transformers](https://huggingface.co/docs/transformers/index) library (you can switch the model type with the CLI param `--model-type`).

## Prerequisites

- [poetry](https://python-poetry.org/docs/) (e.g. `apt install python3-poetry` or `brew install poetry`)

## Running instructions

To run `labeler`, first install the prerequisites, then run the tool with `poetry run python labeler/cli/main.py`. You will get more detailed running instructions with the `--help` command line flag, but in short there are two modes:

- `train`: train a multi-label classifier model (i.e. fine tune a transformer model type of interest for multi-label classification, with `--model-type` used for passing the model type in case you want to use something other than the default `bert-base-uncased`).
- `infer`: make predictions with a trained model.

## Input data for training

The requirements for input data to `labeler` are:

1. A CSV file with two columns:

- `path` (a relative path from the CSV file to the location of the file in question)
- `labels` (a space separated list of labels).

2. Files containing the actual text data (i.e. the files that the `path` field in the above-mentioned CSV points at).

You can find an example set of data under `tests/fixtures`.

## How to train a model, use it for inference

A trivial example case problem is provided in the repo under `tests/fixtures/doctype` with a bunch of HTML, Markdown JSON, YAML files labeled in `tests/fixtures/doctype/labels.csv` with their respective file types, with some instances of the different types of files including content about… mammals (yes, mammals) and not mammals (i.e. the files can be classified as either relating to mammals or not, this being another classification dimension beyond the file types).

For reference, the contents of `tests/fixtures/doctype/labels.csv` look like as follows:

```csv
path;labels
./html/0.html;html
./html/1.html;html
./html/2.html;html
./html/3.html;html
./html/4.html;html
…
./html/aye-aye.html;html mammal
./html/fennec_fox.html;html mammal
./html/platypus.html;html mammal
./html/red_panda.html;html mammal
./html/slow_loris.html;html mammal
./md/0.md;md
./md/1.md;md
./md/2.md;md
./md/3.md;md
…
./md/dolphin.md;md mammal
./md/giraffe.md;md mammal
./md/platypus.md;md mammal
…
./yaml/0.yaml;yaml
./yaml/1.yaml;yaml
./yaml/2.yaml;yaml
…
./yaml/african_elephant.yaml;yaml mammal
./yaml/blue_whale.yaml;yaml mammal
./yaml/giraffe.yaml;yaml mammal
…
./json/0.json;json
./json/1.json;json
./json/2.json;json
…
./json/giraffe.json;json mammal
./json/more_mammals.json;json mammal
./json/more_and_more_mammals.json;json mammal
```

To build the model, run the following command in the root of the repository:

```bash
TOKENIZERS_PARALLELISM=false \
poetry run python labeler/cli/main.py --verbose train \
--labels tests/fixtures/doctype/labels.csv \
--logs ./tests/fixtures/doctype \
--save tests/fixtures/doctype.model \
--batch-size 4 --epochs 100
```

This will output training related summary stats, checkpointing the model every few seconds and eventually saving the best model achieved during the 100 epochs.

To make predictions with the model that was produced by above command at `test/fixtures/doctype.model`, run the following command in the root of the repository:

```bash
poetry run python labeler/cli/main.py --verbose infer \
--logs ./tests/fixtures/doctype/md \
--load tests/fixtures/doctype.model \
--format json \
--threshold 0.5 \
--filter true
```

This will produce output like follows (use `--format csv` if you need a tabular output instead):

```json
[
  {
    "path": "tests/fixtures/doctype/md/5.md",
    "labels": "md",
    "probabilities": {
      "html": 0.3720129728317261,
      "json": 0.2977072596549988,
      "mammal": 0.37781280279159546,
      "md": 0.5835364460945129,
      "yaml": 0.3477823734283447
    }
  },
  {
    "path": "tests/fixtures/doctype/md/1.md",
    "labels": "md",
    "probabilities": {
      "html": 0.3393593728542328,
      "json": 0.2965734899044037,
      "mammal": 0.3827342391014099,
      "md": 0.5878880023956299,
      "yaml": 0.37681934237480164
    }
  },
  {
    "path": "tests/fixtures/doctype/md/0.md",
    "labels": "md",
    "probabilities": {
      "html": 0.36037471890449524,
      "json": 0.2977195978164673,
      "mammal": 0.376987487077713,
      "md": 0.560240626335144,
      "yaml": 0.3395785093307495
    }
  },
  {
    "path": "tests/fixtures/doctype/md/4.md",
    "labels": "md",
    "probabilities": {
      "html": 0.3500433564186096,
      "json": 0.2909952998161316,
      "mammal": 0.3694528043270111,
      "md": 0.5649592876434326,
      "yaml": 0.37255150079727173
    }
  }
]
```

## How to work with the weebl log analysis & bug reporting

Another example problem provided in the repository is a log analysis and bug triaging problem: `tests/fixtures/weebl/weebl_training_data.tar.gz` contains examples of

1. Install `git-lfs` following [installation instructions](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md).

2. `git lfs install` in the root of the repository (once).

3. Preprocess the input data:

```bash
poetry run python tests/fixtures/weebl/process.py \
-i tests/fixtures/weebl/weebl_training_data.tar.gz \
-o tests/fixtures/weebl/processed \
-c tests/fixtures/weebl/processed/labels.csv
```

3. Train a model:

```bash
TOKENIZERS_PARALLELISM=false \
poetry run python labeler/cli/main.py --verbose train \
--labels tests/fixtures/weebl/processed/labels.csv \
--logs ./tests/fixtures/weebl/processed \
--save tests/fixtures/weebl.model \
--batch-size 2 \
--epochs 80
```

# Training an anomaly detection model, Label Studio assisted

You'll find a helper tool under `uploader/cli/upload.py` that allows uploading text data to a Label Studio instance (with which it is the intention to provide a training loop).

```bash
poetry run uploader/cli/upload.py <input_file1> <input_file2> ... -a <auth> -lh <host> -p <proj> [-s <size>] [-o <overlap>] [-b <max_bytes>]
```

Replace <input_file1>, <input_file2>, <auth>, <host>, <proj>, <size>, and <overlap> with the appropriate values.

- <input_file1>, <input_file2>, ...: The input files to be uploaded.
- <auth>: The authorization token for Label Studio.
- <host>: The Label Studio hostname.
- <proj>: The project ID in Label Studio.
- <size> (optional): The size of the batch in number of lines (default: 512).
- <overlap> (optional): The number of overlapping lines between batches (default: 0).
- <max_bytes> (optional): The max # of bytes per batch

The script will upload the input files to the specified Label Studio instance, batching the data in windows of the defined line count and max byte count per batch and overlapping lines. The batch size is pretty approximately enforced still (data is not truncated at the time of writing in the middle of the line for example).

# log-parser

There is also a log parsing utility included as a command-line tool and as a FastAPI deployable web service.
