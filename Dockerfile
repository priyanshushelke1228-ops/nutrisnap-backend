FROM python:3.11-slim

WORKDIR /app

# Zero external pip dependencies needed for pure Python server
COPY . /app

EXPOSE 8080

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD ["python3", "server.py"]
