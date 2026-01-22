#!/bin/bash
set -e  # Para que falle si algo sale mal

# Crear directorio logs si no existe
mkdir -p /app/logs

echo "🚀 Iniciando AgroSync + MeteoTask..."

# 1. Inicia meteoTask.py EN SEGUNDO PLANO (no bloquea)
python -u /app/app/ProgramedJobs/meteoTask.py &

# 2. Guarda PID del meteoTask para poder matarlo después si hace falta
echo $! > /tmp/meteo.pid

# 3. Espera 3 segundos a que se inicie bien
sleep 3

# 4. Inicia Flask en PRIMER PLANO (CMD original)
echo "🌤️ MeteoTask corriendo en background cada 15min"
echo "🔥 Iniciando Flask en puerto 8282..."
exec python -u app/main.py
