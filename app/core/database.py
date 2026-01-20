import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Conexión escalable a PostgreSQL"""
    print("entra en get_db_connection()")
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'host.docker.internal'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'AgroSync'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '12345678')
    )


