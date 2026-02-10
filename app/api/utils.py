import requests
import os

def convertirArrayCoordenadasEnPoligono(coords):
    """
    Convierte [[lat1, lon1], [lat2, lon2], ...] a 'POLYGON((lon1 lat1, lon2 lat2, ...))'
    
    Args:
        coords: list de [lat, lon] como lo recibe de fullstack
    
    Returns:
        str en formato WKT para Auravant API
    """
    if not coords or len(coords) < 3:
        raise ValueError("El polígono necesita al menos 3 coordenadas")
    
    # Extrae lon lat (invierte orden) y formatea
    pairs = [f"{lon} {lat}" for lat, lon in coords]
    
    # Asegura cierre (primer punto == último)
    if pairs[0] != pairs[-1]:
        pairs.append(pairs[0])
    
    return f"POLYGON(({', '.join(pairs)}))"