FROM python:3.10

WORKDIR /app

COPY poetry.lock pyproject.toml /app/

RUN pip install poetry
RUN poetry config virtualenvs.create false

RUN poetry install --only main

COPY ./models--cis-lmu--glotlid /root/.cache/huggingface/hub/models--cis-lmu--glotlid

COPY download.py .
RUN python download.py

COPY app.py .

# USER nonroot:nonroot

ENTRYPOINT ["fastapi", "run", "app.py", "--port", "6776"]
