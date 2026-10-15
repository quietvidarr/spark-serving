FROM python:3.12-slim
RUN pip install --no-cache-dir "aiohttp>=3.9,<4" && command -v nsenter
WORKDIR /app
COPY router.py /app/router.py
CMD ["python", "-u", "/app/router.py"]
