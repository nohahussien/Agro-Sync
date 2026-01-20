import requests
import os
from flask import Blueprint, jsonify

auth_bp = Blueprint('auth', __name__, url_prefix='/agrosync-api')

def getToken():  # ← Ahora SÍ devuelve STRING
    userdata = {
        "username": os.getenv('AURAVANT_AUTH_USER', ''),
        "password": os.getenv('AURAVANT_AUTH_PASS', '')
    }
    urlAuraAuth = os.getenv('AURAVANT_AUTH_URL', 'https://api.auravant.com/api/') + 'auth'
    
    resp = requests.post(urlAuraAuth, data=userdata)
    if resp.status_code == 200:
        return resp.json().get("token")
    return None

@auth_bp.route('/authtoken', methods=['POST'])
def authtoken():
    token = getToken()
    if not token:
        return jsonify({"success": False, "message": "Credenciales inválidas"}), 401
    return jsonify({"success": True, "token": token}), 200
