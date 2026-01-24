from flask import Blueprint, request, jsonify
import requests
from .agro_engine import AgroEngine
from models.field import getParcelas4HistMeteo
from .meteo import getForcasByLatLong
import traceback

import ast

# 1. Definimos el Blueprint
# El nombre 'plant_api' es interno para Flask.
plant_bp = Blueprint('plant_api',  __name__, url_prefix='/agrosync-api')

# 2. Inicializamos el motor
# Se hace aquí para que al importar este archivo, el motor arranque.
try:
    print("🌿 [plant.py] Inicializando Motor AgroEngine...")
    engine = AgroEngine()
    print("✅ [plant.py] Motor listo.")
except Exception as e:
    print(f"❌ [plant.py] Error cargando el motor: {e}")
    engine = None

# --- RUTAS ---

@plant_bp.route('/analyze', methods=['POST'])
def analyze_plant_data():
    """
    Endpoint principal. Recibe una imagen y devuelve análisis unificado.
    Ruta esperada: POST /agrosync-api/analyze (depende del prefix en el main)
    """

    id_parcela = request.args.get('id_parcela', 'default')

    parcela = getParcelas4HistMeteo(id_parcela)
    lat = -1
    lon = -1
    for row in parcela:
        coords = row["coordinates_parcel"]
        coords_list = ast.literal_eval(coords)
        lat=coords_list[0][0]
        lon=coords_list[0][1]

    resultForeCast = getForcasByLatLong(lat, lon)

    print(type(resultForeCast))
    print("Miguel Forecast en Plant: ", resultForeCast)


# Verificaciones de seguridad básicas
    if not engine:
        return jsonify({"error": "El motor de IA no está disponible en este momento"}), 503

    # --- LÓGICA DE ENTRADA (ARCHIVO O URL) ---
    image_bytes = None

    # 1. Intentamos leer archivo directo
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            image_bytes = file.read()

    # 2. Si no hay archivo, intentamos leer URL
    if image_bytes is None:
        image_url = request.form.get('image_url') or (request.json.get('image_url') if request.is_json else None)
        
        if image_url:
            try:
                print(f"⬇️ Descargando imagen desde URL: {image_url}")
                resp = requests.get(image_url, timeout=10)
                if resp.status_code == 200:
                    image_bytes = resp.content
                else:
                    return jsonify({"error": f"Fallo al descargar URL. Status: {resp.status_code}"}), 400
            except Exception as e:
                return jsonify({"error": f"Error de conexión: {str(e)}"}), 400

    # 3. Si sigue vacío, error
    if image_bytes is None:
        return jsonify({"error": "No se recibió imagen. Envíe un archivo 'image' o un string 'image_url'"}), 400

    try:
        # Llamamos al motor (AgroEngine)
        results = engine.analyze_full(image_bytes)

        
        print(results)

        # Devolvemos el JSON estándar
        return jsonify({
            "status": "success",
            "source": "plant_module",
            "data": results,
            "forcast": resultForeCast
        }), 200
        
    except Exception as e:
        print(f"❌ Error en endpoint analyze: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@plant_bp.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que este módulo específico está vivo"""
    return jsonify({
        "module": "plant.py", 
        "status": "active", 
        "engine_loaded": engine is not None
    }), 200