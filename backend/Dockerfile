FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir --default-timeout=120 --retries=10 \
	--extra-index-url https://download.pytorch.org/whl/cpu \
	-r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]