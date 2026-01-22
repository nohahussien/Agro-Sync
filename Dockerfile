FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# PRIMERO copia TODO el código (incluye ProgramedJobs/)
COPY . .

# AHORA el entrypoint (después de COPY . .)
RUN mkdir -p logs
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENV PYTHONPATH=/app
EXPOSE 8282
ENTRYPOINT ["./docker-entrypoint.sh"]
