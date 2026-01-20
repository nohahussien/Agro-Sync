import requests
import os
from flask import request, jsonify, Blueprint

# Blueprint para modularidad
auth_bp = Blueprint('auth', __name__, url_prefix='/agrosync-api')


@auth_bp.route('/authtoken', methods=['POST'])
def default_login():
    try:
             
        # Llamada a Auravant
        userdata = {
            "username": os.getenv('AURAVANT_AUTH_USER', ''),
            "password": os.getenv('AURAVANT_AUTH_PASS', '')
        }

        try:
            resp = requests.post(os.getenv('AURAVANT_AUTH_URL', ''), data=userdata)
        except requests.RequestException:
            return jsonify({"error": "Error comunicando con Auravant"}), 502

        # Si Auravant devuelve error
        if resp.status_code != 200:
            return jsonify({
                "success": False,
                "message": "Credenciales inválidas",
                "auravant_response": resp.json()
            }), 401
        
        body = resp.json()
        token = body.get("token")  # el token viene en la clave "token" [web:9]

        if not token:
            return jsonify({
                "success": False,
                "message": "No se recibió token desde Auravant",
                "auravant_response": body
            }), 502

        # Devuelves el token al front
        return jsonify({
            "success": True,
            "token": token
        }), 200
            
    except Exception as e:
        return jsonify({'error': f'Miguel: Error del servidor: {str(e)}'}), 500
