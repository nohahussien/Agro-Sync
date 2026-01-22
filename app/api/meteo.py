import requests
import pandas as pd
from flask import Blueprint, request, jsonify
from models.field import getParcelas4HistMeteo

meteo_bp = Blueprint('meteo', __name__, url_prefix='/agrosync-api')

@meteo_bp.route('/forecast', methods=['GET'])
def current_weather():
    # TRAE EL ACTUAL SEGUN PARÁMETROS DE ENTRADA DE LONGITUD Y LATITUD
    print("entro en forecast")
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m,wind_direction_10m",
        "timezone": "auto"
    }

    print("entro 2")

    r = requests.get(url, params=params)

    print("entro 3")

    r.raise_for_status()

    print("entro 3.4")

    data = r.json()

    cur = data["current"]

    print("entro 4")

    df = pd.DataFrame([{
        "time": pd.to_datetime(cur["time"]),
        "temperature": cur["temperature_2m"],
        "relative_humidity": cur["relative_humidity_2m"],
        "precipitation": cur["precipitation"],
        "cloud_cover": cur["cloud_cover"],
        "wind_speed": cur["wind_speed_10m"],
        "wind_direction": cur["wind_direction_10m"]
    }])

    return df.to_dict(orient="records")
