FROM python:3.12-slim
WORKDIR /app

# Requirements first: this layer is cached until the dependency list changes,
# so editing engine code does not trigger a full reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home aegis && chown -R aegis:aegis /app
USER aegis

# Cached price data lives here; mounting a volume over it makes the container
# survive restarts without re-fetching from yfinance.
VOLUME ["/app/data/store"]

CMD ["python", "-m", "publisher.report"]
