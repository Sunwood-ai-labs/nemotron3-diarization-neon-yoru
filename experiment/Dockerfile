FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libgomp1 curl ca-certificates ffmpeg build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir 'nemo_toolkit[asr]==3.0.0' numpy scipy soundfile 'huggingface_hub>=1.0' faster-whisper
COPY runtime/linux-release.tar.gz runtime/linux-release.tar.gz.sha256 /tmp/
RUN cd /tmp && sha256sum -c linux-release.tar.gz.sha256 && mkdir /opt/nemo && tar xzf linux-release.tar.gz --strip-components=1 -C /opt/nemo
ENV PATH="/opt/nemo/bin:${PATH}" LD_LIBRARY_PATH="/opt/nemo/lib" PYTHONUNBUFFERED=1 HF_HOME=/work/models/hf-cache OMP_NUM_THREADS=8
WORKDIR /work
CMD ["python", "experiment.py"]
