FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    curl

RUN pip3 install poetry

WORKDIR /app
COPY ./labeler /app
COPY ./poetry.lock /app
COPY ./pyproject.toml /app

RUN poetry install --no-dev

ENV API_KEY=invalid
ENV DRAIN_STATE_PATH="/data/drain3_state.json"

EXPOSE 8080

CMD ["poetry", "run", "uvicorn", "labeler.parser.service:app", "--host", "0.0.0.0", "--port", "8080"]
