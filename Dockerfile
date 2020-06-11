FROM python:3.6-slim

WORKDIR /app

COPY op /app

CMD ["python", "hello.py"]
