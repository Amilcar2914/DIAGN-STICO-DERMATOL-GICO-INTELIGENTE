from flask import Flask, request, jsonify, send_from_directory
import hashlib
import io
import os
import warnings

import joblib
import numpy as np
from PIL import Image

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__, static_folder='../public', static_url_path='')


@app.route('/')
def index():
    return app.send_static_file('index.html')


# Cargar modelo entrenado
try:
    model_path = os.path.join(os.path.dirname(__file__), 'modelo_derma.pkl')
    imputer_path = os.path.join(os.path.dirname(__file__), 'imputer.pkl')
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    print("[init] Modelo e imputer cargados correctamente")
except Exception as e:  # pragma: no cover - log en tiempo de arranque
    model = None
    imputer = None
    print(f"[init] Error cargando modelo: {e}")

class_mapping = {
    1: "Psoriasis",
    2: "Dermatitis Seborreica",
    3: "Liquen Plano",
    4: "Pitiriasis Rosada",
    5: "Eczema Cronico",
    6: "Pitiriasis Rubra Pilaris",
}

# Referencias locales para comparacion rapida
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), 'imagen_refrencia')
REFERENCE_FILES = {
    "Psoriasis": "Psoriasis 1.png",
    "Pitiriasis Rosada": "Pitiriasis Rosada.png",
    "Liquen Plano": "Liquen Plano.png",
}
MAX_CACHE_ITEMS = 30
image_cache = {}


def build_signature(img_array, bins=24):
    """Crea una firma compacta basada en histogramas de color normalizados."""
    channels = []
    for channel in range(3):
        hist, _ = np.histogram(
            img_array[:, :, channel].flatten(),
            bins=bins,
            range=(0, 255),
            density=True,
        )
        channels.append(hist)
    signature = np.concatenate(channels)
    norm = np.linalg.norm(signature)
    if norm == 0:
        return signature
    return signature / norm


def similarity_score(sig_a, sig_b):
    denom = float(np.linalg.norm(sig_a) * np.linalg.norm(sig_b))
    if denom == 0:
        return 0.0
    return float(np.dot(sig_a, sig_b) / denom)


def load_reference_signatures():
    signatures = {}
    for label, filename in REFERENCE_FILES.items():
        path = os.path.join(REFERENCE_DIR, filename)
        if not os.path.exists(path):
            print(f"[warn] Imagen de referencia ausente: {path}")
            continue
        try:
            with Image.open(path) as ref_img:
                resized = ref_img.convert("RGB").resize((300, 300))
                signatures[label] = build_signature(np.array(resized))
        except Exception as exc:  # pragma: no cover - log de recursos faltantes
            print(f"[warn] No se pudo cargar {filename}: {exc}")
    return signatures


reference_signatures = load_reference_signatures()


def cache_result(image_hash, payload):
    image_cache[image_hash] = payload
    if len(image_cache) > MAX_CACHE_ITEMS:
        first_key = next(iter(image_cache))
        image_cache.pop(first_key, None)


# --- ENDPOINT PARA ANALIZAR IMAGEN ---
@app.route('/api/analyze_image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    try:
        # 1. Leer imagen y hash para cache ligera (no se guarda la imagen)
        image_bytes = file.read()
        image_hash = hashlib.md5(image_bytes).hexdigest()

        if image_hash in image_cache:
            cached_payload = image_cache[image_hash].copy()
            cached_payload["cached"] = True
            return jsonify({"status": "success", **cached_payload})

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((400, 400))
        img_array = np.array(img)

        # Firma de imagen para comparar con las referencias locales
        signature = build_signature(img_array)
        match_breakdown = []
        for label, ref_sig in reference_signatures.items():
            score = similarity_score(signature, ref_sig)
            match_breakdown.append({
                "label": label,
                "score": round(score * 100, 2)
            })
        match_breakdown = sorted(match_breakdown, key=lambda x: x["score"], reverse=True)

        # A) Analisis de color
        r_channel = img_array[:, :, 0].astype(float)
        g_channel = img_array[:, :, 1].astype(float)
        b_channel = img_array[:, :, 2].astype(float)

        r_mean = np.mean(r_channel)
        g_mean = np.mean(g_channel)
        b_mean = np.mean(b_channel)

        redness_score = r_mean - (g_mean + b_mean) / 2
        purple_score = (r_mean + b_mean) / 2 - g_mean
        blue_vs_green = b_mean - g_mean

        is_lichen_planus = False
        if purple_score > 8 and blue_vs_green > 3:
            is_lichen_planus = True
            erythema = 2
        elif redness_score > 50:
            erythema = 3
        elif redness_score > 30:
            erythema = 2
        elif redness_score > 12:
            erythema = 1
        else:
            erythema = 0

        # B) Textura y descamacion
        gray = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel

        texture_score = np.std(gray)

        from scipy.ndimage import sobel  # import local para reducir carga de arranque
        gradient_x = sobel(gray, axis=0)
        gradient_y = sobel(gray, axis=1)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        edge_score = np.mean(gradient_magnitude)

        if texture_score > 60 and edge_score > 20:
            scaling = 3
        elif texture_score > 45 or edge_score > 15:
            scaling = 2
        elif texture_score > 25 or edge_score > 8:
            scaling = 1
        else:
            scaling = 0

        if is_lichen_planus:
            scaling = min(scaling, 1)

        # C) Bordes
        from scipy.ndimage import gaussian_filter
        gray_smooth = gaussian_filter(gray, sigma=2)
        edges = np.abs(sobel(gray_smooth, axis=0)) + np.abs(sobel(gray_smooth, axis=1))
        edge_strength = np.percentile(edges, 95)

        if edge_strength > 50 or is_lichen_planus:
            borders = 3
        elif edge_strength > 30:
            borders = 2
        elif edge_strength > 15:
            borders = 1
        else:
            borders = 0

        # D) Distribucion y conteo de lesiones
        from scipy.ndimage import label
        binary = edges > np.percentile(edges, 75)
        _, num_features = label(binary)

        # E) Decision final combinando patrones clinicos
        itching = 1
        detected_condition = None

        if is_lichen_planus or (purple_score > 5 and num_features >= 3):
            detected_condition = "Liquen Plano"
            erythema = 2
            scaling = 0
            itching = 3
            borders = 3
        elif scaling >= 2 and edge_strength > 40 and texture_score > 40:
            detected_condition = "Psoriasis"
            erythema = max(erythema, 2)
            scaling = max(scaling, 2)
            itching = 2
            borders = 3
        elif num_features <= 3 and erythema <= 2 and scaling <= 2:
            detected_condition = "Pitiriasis Rosada"
            erythema = min(erythema, 2)
            scaling = min(scaling, 1)
            itching = 1
            borders = 2
        elif redness_score > 25:
            detected_condition = "Psoriasis" if scaling >= 2 else "Pitiriasis Rosada"
            if scaling >= 2:
                erythema = 2
                scaling = 2
                itching = 2
                borders = 3
            else:
                erythema = 2
                scaling = max(scaling, 1)
                itching = 1
                borders = 2
        else:
            detected_condition = "Pitiriasis Rosada"
            erythema = max(erythema, 1)
            scaling = max(scaling, 1)
            itching = 1
            borders = 2

        if erythema == 0:
            erythema = 1
        if scaling == 0 and not is_lichen_planus:
            scaling = 1

        # Priorizar coincidencia de referencias si hay match claro
        top_match_label = match_breakdown[0]["label"] if match_breakdown else None
        top_match_score = match_breakdown[0]["score"] if match_breakdown else 0
        if top_match_label and top_match_score >= 70 and not detected_condition:
            detected_condition = top_match_label
        if not detected_condition and top_match_label:
            detected_condition = top_match_label

        # Preparar lista de imagenes similares para el frontend
        similar_images_payload = []
        for match in match_breakdown:
            filename = REFERENCE_FILES.get(match["label"])
            if filename:
                similar_images_payload.append({
                    "label": match["label"],
                    "confidence": match["score"],
                    "url": f"/api/reference_image/{filename}"
                })

        payload = {
            "features": {
                "erythema": int(erythema),
                "scaling": int(scaling),
                "itching": int(itching),
                "borders": int(borders),
            },
            "preliminary_diagnosis": detected_condition,
            "similar_images": similar_images_payload, # Cambiado de 'similarity' a 'similar_images' con URLs
            "cached": False,
        }

        cache_result(image_hash, payload)

        print(f"[analyze_image] Resultado: {payload}")
        return jsonify({"status": "success", **payload})
    except Exception as e:
        print(f"Error en analisis de imagen: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/reference_image/<path:filename>')
def serve_reference_image(filename):
    return send_from_directory(REFERENCE_DIR, filename)


# --- ENDPOINT PREDICCION (Modelo Random Forest) ---
@app.route('/api/predict', methods=['POST'])
def predict():
    if not model or not imputer:
        return jsonify({"error": "Model not loaded", "status": "error"}), 500

    try:
        data = request.get_json()

        erythema = int(data.get('erythema', 0))
        scaling = int(data.get('scaling', 0))
        definite_borders = int(data.get('definite_borders', 0))
        itching = int(data.get('itching', 0))
        age = int(data.get('age', 30))

        features = [0] * 34

        features[0] = erythema
        features[1] = scaling
        features[2] = definite_borders
        features[3] = itching

        # === PATRON LIQUEN PLANO ===
        if itching >= 2 and definite_borders >= 2 and erythema <= 2 and scaling <= 1:
            features[5] = 3
            features[7] = 3
            features[11] = 3
            features[16] = 3
            features[17] = 2
            features[18] = 0
            features[19] = 0
            features[20] = 0
            features[21] = 3
            features[24] = 3
            features[25] = 0
            features[26] = 3
            features[27] = 1
            features[28] = 3
            features[31] = 3
            features[32] = 3
            features[4] = 0
            features[8] = 0
            features[9] = 0

        # === PATRON PSORIASIS ===
        elif erythema >= 2 and scaling >= 2 and definite_borders >= 2:
            features[4] = 3
            features[8] = 3
            features[10] = 1
            features[16] = 3
            features[17] = 3
            features[18] = 3
            features[19] = 3
            features[20] = 3
            features[21] = 0
            features[22] = 3
            features[23] = 3
            features[24] = 0
            features[25] = 3
            features[26] = 0
            features[27] = 0
            features[28] = 0
            features[31] = 3
            features[32] = 0
            features[5] = 0
            features[9] = 2

        # === PATRON PITIRIASIS ROSADA ===
        elif erythema <= 2 and scaling <= 2 and itching <= 2:
            features[4] = 0
            features[5] = 0
            features[8] = 0
            features[9] = 0
            features[10] = 0
            features[15] = 3
            features[16] = 2
            features[17] = 2
            features[18] = 2
            features[19] = 0
            features[20] = 0
            features[21] = 0
            features[22] = 0
            features[23] = 0
            features[24] = 0
            features[25] = 0
            features[26] = 0
            features[27] = 2
            features[28] = 0
            features[31] = 2
            features[32] = 0

        # === PATRON DERMATITIS SEBORREICA ===
        elif scaling >= 1 and erythema <= 1:
            features[9] = 3
            features[4] = 0
            features[5] = 0
            features[17] = 2
            features[18] = 2
            features[27] = 2
            features[31] = 2

        # === PATRON ECZEMA CRONICO ===
        elif itching >= 3 and erythema >= 2:
            features[12] = 3
            features[15] = 3
            features[16] = 3
            features[17] = 2
            features[27] = 3
            features[31] = 3

        # === PATRON PITIRIASIS RUBRA PILARIS ===
        elif scaling >= 3:
            features[6] = 3
            features[17] = 3
            features[29] = 3
            features[30] = 3
            features[31] = 2

        if features[13] == 0:
            features[13] = 1 if erythema >= 2 else 0
        if features[14] == 0:
            features[14] = 1

        features[33] = age

        features_array = np.array([features])
        features_imputed = imputer.transform(features_array)

        pred = int(model.predict(features_imputed)[0])

        try:
            probs = model.predict_proba(features_imputed)[0]
            conf = round(np.max(probs) * 100, 2)
            
            # Crear lista de probabilidades ordenada
            probabilities_list = []
            for i, prob in enumerate(probs):
                class_name = class_mapping.get(i + 1, f"Clase {i+1}")
                probabilities_list.append({
                    "label": class_name,
                    "confidence": round(prob * 100, 2)
                })
            
            # Ordenar de mayor a menor confianza
            probabilities_list.sort(key=lambda x: x["confidence"], reverse=True)
            
        except Exception:
            conf = 0
            probabilities_list = []

        return jsonify({
            "status": "success",
            "diagnosis": class_mapping.get(pred, "Desconocido"),
            "confidence": conf,
            "probabilities": probabilities_list
        })
    except Exception as e:
        print(f"Error en prediccion: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
