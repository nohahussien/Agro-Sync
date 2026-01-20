import requests
import base64
import os
from flask import request, jsonify, Blueprint

# --- CONFIGURACIÓN DEL BLUEPRINT ---
plant_bp = Blueprint('plant', __name__, url_prefix='/agrosync-api')

ENDPOINT = "https://api.plant.id/v3/identification"

# --- FUNCIONES AUXILIARES (Privadas del módulo) ---
def _parser_backend(api_response):
    """Parsea la respuesta cruda de Plant.id a tu formato simplificado."""
    if not api_response or 'result' not in api_response:
        return None

    result = api_response['result']
    parsed_data = {}

    # 1. Identificación
    classification = result.get('classification', {}).get('suggestions', [])
    if classification:
        top_match = classification[0]
        parsed_data['plant_name'] = top_match['name']
        parsed_data['plant_confidence'] = top_match['probability']
        
        details = top_match.get('details', {})
        common_names = details.get('common_names', [])
        parsed_data['common_name'] = common_names[0] if common_names else "Desconocido"
        parsed_data['info_url'] = details.get('url', None) 
    
    # 2. Salud
    is_healthy = result.get('is_healthy', {})
    parsed_data['healthy_prob'] = is_healthy.get('probability', 0)

    # 3. Enfermedades
    disease_obj = result.get('disease', {})
    disease_suggestions = disease_obj.get('suggestions', [])
    
    parsed_data['diseases'] = []
    if disease_suggestions:
        for d in disease_suggestions:
            if d['probability'] > 0.10: 
                disease_info = {
                    'name': d['name'],
                    'probability': d['probability'],
                    'description': d.get('details', {}).get('description', 'Sin descripción.')[:200]
                }
                parsed_data['diseases'].append(disease_info)

    return parsed_data

# --- RUTAS ---

@plant_bp.route('/analyze', methods=['POST'])
def analyze_plant():
    try:
        # 1. Validación de entrada (Imagen)
        if 'image' not in request.files:
            return jsonify({"success": False, "message": "No se recibió imagen"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "message": "Archivo vacío"}), 400

        # 2. Procesamiento de imagen a Base64
        image_bytes = file.read()
        base64_img = base64.b64encode(image_bytes).decode("utf-8")

        # 3. Preparar Payload para Plant.id
        # Usamos os.getenv para la seguridad, igual que en tu ejemplo de Auth
        api_key = os.getenv('PLANT_ID_KEY', '') 
        
        headers = {
            "Content-Type": "application/json",
            "Api-Key": api_key
        }
        
        data = {
            "images": [base64_img],
            "latitude": 40.4168, # Podrías recibirlas en el request.form si quisieras
            "longitude": -3.7038,
            "health": "all",
            "similar_images": True
        }

        # 4. Llamada a API Externa
        try:
            resp = requests.post(ENDPOINT, headers=headers, json=data)
        except requests.RequestException:
            return jsonify({"error": "Error comunicando con Plant.id"}), 502

        # 5. Manejo de respuesta de la API externa
        if resp.status_code != 200 and resp.status_code != 201:
            return jsonify({
                "success": False, 
                "message": "Error del proveedor de IA",
                "provider_response": resp.text
            }), resp.status_code

        raw_json = resp.json()
        
        # 6. Parsing y Respuesta final
        clean_data = _parser_backend(raw_json)
        
        if not clean_data:
            return jsonify({"success": False, "message": "No se pudo interpretar la respuesta de la IA"}), 500

        return jsonify({
            "success": True,
            "data": clean_data
        }), 200

    except Exception as e:
        # Tu formato de error personalizado
        return jsonify({'error': f' Error del servidor: {str(e)}'}), 500