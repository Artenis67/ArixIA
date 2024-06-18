FROM python:3.9-slim

WORKDIR /flowerColorIA

COPY . /flowerColorIA/

CMD ["python3", "flowerColorIA.py"]
