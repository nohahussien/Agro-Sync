from sqlalchemy import Column, Integer, String  
from app.core.database import get_db_connection
import psycopg2
from psycopg2.extras import RealDictCursor

def getParcelas4HistMeteo():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT uid_parcel, coordinates_parcel
                FROM parcels
            """)
            parcelas = cur.fetchall()  # ← Recupera TODAS las filas
            return parcelas  # lista de dicts

    except Exception as e:
        conn.rollback()
        print(f"Error en getParcelas4HistMeteo: {e}")
        return []
    finally:
        conn.close()