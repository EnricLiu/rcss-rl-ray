FROM rayproject/ray:2.52.0-py312

WORKDIR /app

COPY . .

RUN uv sync
