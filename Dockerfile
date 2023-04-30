FROM ubuntu:22.04
ENV API_KEY=invalid
ENV DRAIN_STATE_PATH="/data/drain3_state.json"

RUN apt-get update && apt-get install -y \
    python3-pip \
    curl

RUN pip3 install poetry

WORKDIR /app
COPY ./poetry.lock /app/poetry.lock
COPY ./pyproject.toml /app/pyproject.toml

RUN poetry install --no-dev

COPY ./labeler /app/labeler

EXPOSE 8080

CMD ["poetry", "run", "uvicorn", "labeler.parser.service:app", "--host", "0.0.0.0", "--port", "8080"]
