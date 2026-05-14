FROM python:3.11-slim

WORKDIR /app

# System dependencies for some python packages if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code as fallback (volume mount will override at runtime)
COPY . .

# Default command is bash so you can run any script interactively
CMD ["bash"]