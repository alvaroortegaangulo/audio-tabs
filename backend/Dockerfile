FROM python:3.10-slim

WORKDIR /app

# Dependencias sistema: ffmpeg + libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg libsndfile1 lilypond libgomp1 \
  && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir numpy==1.26.4 Cython==0.29.37
RUN pip install --no-cache-dir -r /app/requirements.txt

# Parches para madmom: compatibilidad con Python 3.10+ y NumPy 1.20+
COPY patch_madmom.py /tmp/patch_madmom.py
RUN python3 /tmp/patch_madmom.py && rm /tmp/patch_madmom.py

# Demucs/torch para separación de stems
COPY requirements-demucs.txt /app/requirements-demucs.txt
RUN pip install --no-cache-dir -r /app/requirements-demucs.txt

COPY app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
