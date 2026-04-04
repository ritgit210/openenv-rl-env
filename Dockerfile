FROM python:3.11-slim

WORKDIR /app

# Install project dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic httpx pillow python-multipart openenv-core

# Create a package directory to ensure relative imports like '..models' work
COPY . cosmic_bytes/

# Ensure the root is in PYTHONPATH
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=7860

EXPOSE 7860

# Run the server using the module path
CMD ["uvicorn", "cosmic_bytes.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
