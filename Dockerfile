FROM python:3.6

WORKDIR /app

COPY op /app

CMD ["python", "hello.py"]
