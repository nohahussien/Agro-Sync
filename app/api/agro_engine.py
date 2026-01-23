import cv2
import numpy as np
import math
from roboflow import Roboflow

# --- CONFIGURACIÓN ---
ROBOFLOW_API_KEY = "Meni7XPRgKOEkHJeXRHz"
MIRROR_BACK_LENS = True  

class AgroEngine:
    def __init__(self):
        print("⚙️ [AgroEngine] Cargando modelo Roboflow...")
        self.rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        self.project = self.rf.workspace("agridrone-pblcc").project("agridetect")
        self.version = self.project.version(3)
        self.model = self.version.model
        print("✅ [AgroEngine] Modelo cargado.")

    def _decode_bytes_to_numpy(self, image_bytes):
        """Decodifica bytes de imagen a array numpy"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"❌ Error decodificando imagen: {e}")
            return None

    # --- ALGORITMOS DE VISIÓN (Refactorizados de test_roboflow.py) ---
    
    def _find_green_centroid(self, img_full, x_box, y_box, w_box, h_box):
        """
        Calcula el CENTROIDE (Centro de gravedad) de la masa verde dentro de una detección.
        """
        # Recortar ROI (Region of Interest)
        y_start, y_end = int(y_box - h_box/2), int(y_box + h_box/2)
        x_start, x_end = int(x_box - w_box/2), int(x_box + w_box/2)
        h_img, w_img, _ = img_full.shape
        
        # Validar límites
        y_start, y_end = max(0, y_start), min(h_img, y_end)
        x_start, x_end = max(0, x_start), min(w_img, x_end)
        
        roi = img_full[y_start:y_end, x_start:x_end]
        if roi.size == 0: return int(x_box), int(y_box) # Fallback al centro de la caja

        # Detectar Verde (HSV)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_roi, lower_green, upper_green)
        
        # Momentos de imagen
        M = cv2.moments(mask)
        
        if M["m00"] != 0:
            # Fórmulas del centroide: Cx = M10/M00, Cy = M01/M00
            cX_local = int(M["m10"] / M["m00"])
            cY_local = int(M["m01"] / M["m00"])
            # Retornar coordenadas globales
            return int(x_start + cX_local), int(y_start + cY_local)
        else:
            # Si no hay verde, devolvemos el centro de la caja original
            return int(x_box), int(y_box)

    def _detect_sky_hsv(self, img_cv2):
        """
        Detecta cielo usando visión por computador clásica (HSV) si la IA falla.
        """
        hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
        h, w, _ = img_cv2.shape
        # Rango azul/blanco brillante para cielo
        mask = cv2.inRange(hsv, np.array([85, 30, 100]), np.array([135, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        
        # Ignorar la mitad inferior para evitar falsos positivos en suelo
        mask[int(h*0.5):, :] = 0
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 5000:
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                # Devolvemos formato compatible con Roboflow
                return [{
                    'class': 'sky', 
                    'confidence': 0.99, 
                    'x': x + w_rect/2, 
                    'y': y + h_rect/2, 
                    'width': w_rect, 
                    'height': h_rect
                }]
        return []

    # --- MATEMÁTICA 3D ---

    def _pixel_to_unit_vector_dual_fisheye(self, x, y, width, height):
        """Calcula vector 3D para cámaras Dual Fisheye (Insta360)."""
        lens_width = width / 2
        lens_radius = height / 2 
        
        if x < lens_width:
            is_front_lens = True
            center_x = lens_width / 2
        else:
            is_front_lens = False
            center_x = lens_width + (lens_width / 2)

        center_y = height / 2
        u = (x - center_x) / lens_radius
        v = (y - center_y) / lens_radius
        distance_sq = u*u + v*v
        
        if distance_sq > 1.0: return None 

        z_component = math.sqrt(max(0, 1.0 - distance_sq))

        vec_x = u
        vec_y = -v 
        
        if is_front_lens:
            vec_z = z_component 
        else:
            vec_z = -z_component
            if MIRROR_BACK_LENS: vec_x = -vec_x # Corrección de espejo lente trasera

        return {"x": round(vec_x, 4), "y": round(vec_y, 4), "z": round(vec_z, 4)}

    def _convert_to_aframe_vector(self, vector_math, radius=10):
        if vector_math is None: return None
        # A-Frame: Y-Up, -Z Forward
        return {
            "x": round(vector_math['x'] * radius, 4),
            "y": round(vector_math['y'] * radius, 4),
            "z": round(vector_math['z'] * -1 * radius, 4)
        }

    # --- LÓGICA PRINCIPAL ---

    def _process_roboflow_hybrid(self, image_numpy):
        try:
            if image_numpy is None: return {"error": "Imagen no válida"}
            height, width, _ = image_numpy.shape
            img_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)

            # 1. Inferencia Roboflow
            response = self.model.predict(img_rgb, confidence=5).json()
            detections = response['predictions'] if 'predictions' in response else []

            # 2. Fallback: Si no hay cielo detectado por IA, usar HSV
            if not any(d['class'] in ["sky", "cielo", "cloud"] for d in detections):
                detections.extend(self._detect_sky_hsv(image_numpy))

            # 3. Definición de Categorías
            categories = {
                "sky": {
                    "labels": ["sky", "cielo", "cloud"],
                    "fallback_vec": {"x": 0.0, "y": 0.7071, "z": 0.7071}, 
                    "best_det": None, "max_conf": 0
                },
                "soil": {
                    "labels": ["field-soil", "unused-land", "soil", "land", "ground"],
                    "fallback_vec": {"x": 0.0, "y": -0.8, "z": 0.6}, 
                    "best_det": None, "max_conf": 0
                },
                "crop": {
                    "labels": ["crop", "trees", "agriculture-land", "plants", "weed"],
                    "fallback_vec": {"x": 0.0, "y": -0.5, "z": 0.866}, 
                    "best_det": None, "max_conf": 0
                }
            }

            # 4. Filtrado de mejores detecciones
            for det in detections:
                cls = det['class'].lower() # Normalizamos a minusculas
                conf = det['confidence']
                for cat_key, cat_data in categories.items():
                    # Comprobamos si la clase está en las etiquetas (también normalizadas)
                    if cls in [l.lower() for l in cat_data["labels"]]:
                        if conf > cat_data["max_conf"]:
                            cat_data["max_conf"] = conf
                            cat_data["best_det"] = det

            # 5. Construcción de Resultados
            final_results = {}
            for cat_key, cat_data in categories.items():
                best_det = cat_data["best_det"]
                px, py = 0, 0
                vector_3d = None
                source = "FALLBACK"
                confidence = 0

                if best_det:
                    source = "IA"
                    confidence = best_det['confidence']
                    
                    if cat_key == "crop":
                        # LÓGICA ESPECIAL: CENTROIDE VERDE
                        px, py = self._find_green_centroid(
                            image_numpy, 
                            best_det['x'], best_det['y'], 
                            best_det['width'], best_det['height']
                        )
                        source = "IA-CENTROID"
                    else:
                        # Centro de la caja normal
                        px, py = int(best_det['x']), int(best_det['y'])
                        if confidence == 0.99 and cat_key == "sky": source = "HSV-CV"

                    vector_3d = self._pixel_to_unit_vector_dual_fisheye(px, py, width, height)

                # Si vector_3d sigue siendo None (no detectado o fuera de lente), usar fallback
                if vector_3d is None:
                    vector_3d = cat_data["fallback_vec"]
                    if source != "FALLBACK": source += "-OUT-OF-LENS" # Detectado pero en zona negra
                
                final_results[cat_key] = {
                    "detected": (confidence > 0),
                    "source": source,
                    "confidence": confidence,
                    "pixel_coords": {"x": px, "y": py},
                    "aframe_position": self._convert_to_aframe_vector(vector_3d, radius=10)
                }
            
            return final_results

        except Exception as e:
            print(f"❌ Error en process_roboflow_hybrid: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def analyze_full(self, raw_image_bytes):
        """Método público API"""
        image_numpy = self._decode_bytes_to_numpy(raw_image_bytes)
        if image_numpy is None: return {"error": "Archivo corrupto o no es imagen"}
        return {"telemetry_roi": self._process_roboflow_hybrid(image_numpy)}