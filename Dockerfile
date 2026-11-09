FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

COPY requirements.txt .
# CPU-only PyTorch avoids pulling unused CUDA libraries into this CPU app.
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

# Bundle the existing retrieval model so replicas never need to download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app.py .
COPY routing/ routing/
COPY data/ data/
COPY .streamlit/config.toml .streamlit/config.toml

# Build the same validated cache used locally; allow runtime regeneration if needed.
RUN python -c "from routing.search import IncidentSearch, DEFAULT_CACHE_PATH; IncidentSearch(); assert DEFAULT_CACHE_PATH.is_file()" \
    && useradd --create-home appuser \
    && chown -R appuser:appuser /app/.cache
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
