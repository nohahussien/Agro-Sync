from flask import Flask
from flask_cors import CORS
from app.api.auth import auth_bp
from app.api.fields import field_bp

app = Flask(__name__)
CORS(app)  # ← ESTO HACE LA MAGIA ✨

# Registrar blueprint
app.register_blueprint(auth_bp)
app.register_blueprint(field_bp)

@app.route('/')
def home():
    return {
        'message': '🎭 AgroSync funcionando cambiado a v1!',
        'endpoints': {
            'POST /agrosync-api/authtoken': 'Header Authentication: (email - password) Auravant'
        }
    }

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8282, debug=True)
