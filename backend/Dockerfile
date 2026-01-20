FROM python:3.11-slim

WORKDIR /app

# Dependencias sistema: ffmpeg + libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 lilypond \
  && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Demucs/torch opcional: si quieres, descomenta en tu build real o usa requirements-demucs.txt.
# COPY requirements-demucs.txt /app/requirements-demucs.txt
# RUN pip install --no-cache-dir -r /app/requirements-demucs.txt

COPY app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
