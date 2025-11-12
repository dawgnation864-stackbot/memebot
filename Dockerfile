FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY memebot.py /app/
# default: simulation on unless you flip it with env vars
ENV SIMULATION_MODE=True
ENV START_MODE=start
CMD ["python", "memebot.py"]
