from flask import Flask
from flask_cors import CORS
from app.api.auth import auth_bp
from app.api.fields import field_bp
from app.api.plant import plant_bp
from app.api.meteo import meteo_bp

app = Flask(__name__)
CORS(app)  # ← ESTO HACE LA MAGIA ✨

# Registrar blueprint
app.register_blueprint(auth_bp)
app.register_blueprint(field_bp)
app.register_blueprint(meteo_bp)
app.register_blueprint(plant_bp)

@app.route('/')
def home():
    return {
        "service": "AgroSync API",
        "version": "v5",
        "description": "API de integración con Auravant y servicios meteorológicos",
        "base_url": "http://localhost:8282/agrosync-api",
        "authentication": {
            "type": "Bearer Token",
            "provider": "Auravant",
            "endpoint": "/agrosync-api/authtoken"
        },
        "endpoints": [
            {
                "path": "/agrosync-api/authtoken",
                "method": "POST",
                "auth_required": False,
                "description": "Obtiene token de autenticación contra Auravant",
                "headers": {
                    "SUBDOMAIN": "string",
                    "EXTENSION_ID": "string",
                    "SECRET": "string"
                },
                "response": {
                    "success": "boolean",
                    "token": "string"
                }
            },
            {
                "path": "/agrosync-api/getfields",
                "method": "POST",
                "auth_required": True,
                "description": "Obtiene la lista de campos del usuario",
                "external_api": "Auravant getfields"
            },
            {
                "path": "/agrosync-api/agregarlote",
                "method": "POST",
                "auth_required": True,
                "description": "Crea un nuevo lote en un campo",
                "body": {
                    "nombrecampo": "string",
                    "shape": "[[lat, lon], [lat, lon], ...]"
                },
                "external_api": "Auravant agregarlote"
            },
            {
                "path": "/agrosync-api/eliminarlote",
                "method": "GET",
                "auth_required": True,
                "query_params": {
                    "lote": "ID del lote"
                },
                "external_api": "Auravant borrarlotes"
            },
            {
                "path": "/agrosync-api/forecast",
                "method": "GET",
                "auth_required": False,
                "description": "Obtiene condiciones meteorológicas actuales",
                "body": {
                    "lat": "float",
                    "lon": "float"
                },
                "external_api": "Open-Meteo"
            }
        ]
    }

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8282, debug=True)
