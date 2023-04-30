FROM python:3.10

WORKDIR /app
COPY . /app

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

RUN poetry install --no-dev

EXPOSE 8080

ENV REDIS_HOST localhost
ENV REDIS_PORT 6379
ENV REDIS_DB 0

CMD ["uvicorn", "log_parser:app", "--host", "0.0.0.0", "--port", "8000"]
