# labeler - ML driven labeler of files

A transformer model based tool for associating textual data with one or more labels (multi-label classifier of text data).

## Prerequisites

- [poetry](https://python-poetry.org/docs/) (e.g. `apt install python3-poetry` or `brew install poetry`)

## Example

To run labeler, first install the prerequisites, then run the tool with `poetry run python labeler/cli/main.py`.

A trivial example case problem is provided in the repo under `tests/fixtures/doctype` with a bunch of HTML, Markdown JSON, YAML files labeled in `tests/fixtures/doctype/labels.csv` with their respective file types. The requirements for input data are:

- A CSV file with two columns: `path` (a relative path from the CSV file to the location of the file in question) and `labels` (a space separated list of labels).
- Files that contain the actual text data (i.e. the files that the `path` field in the above-mentioned CSV points at).

```bash
poetry install
poetry run python labeler/cli/main.py train \
--labels tests/fixtures/doctype/labels.csv \
--logs ./tests/fixtures/doctype \
--save tests/fixtures/doctype.model
```
