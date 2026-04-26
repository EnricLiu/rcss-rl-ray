FROM rayproject/ray:2.52.0-py312

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
