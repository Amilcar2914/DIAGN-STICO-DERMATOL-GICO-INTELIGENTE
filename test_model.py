"""
Script de prueba para verificar el modelo
"""
import joblib
import numpy as np

print("Cargando modelo e imputer...")
model = joblib.load('modelo_derma.pkl')
imputer = joblib.load('imputer.pkl')

print(f"✅ Modelo cargado: {type(model)}")
print(f"✅ Imputer cargado: {type(imputer)}")

# Probar con valores de ejemplo
features = [0] * 34
features[0] = 3  # erythema alto
features[1] = 3  # scaling alto
features[2] = 2  # borders
features[3] = 2  # itching
features[33] = 35  # age

print(f"\n📊 Input features shape: {np.array([features]).shape}")

# Aplicar imputer
features_imputed = imputer.transform(np.array([features]))
print(f"📊 After imputer shape: {features_imputed.shape}")

# Predecir
pred = int(model.predict(features_imputed)[0])
probs = model.predict_proba(features_imputed)
conf = round(np.max(probs) * 100, 2)

class_mapping = {
    1: "Psoriasis", 2: "Dermatitis Seborreica", 3: "Liquen Plano",
    4: "Pitiriasis Rosada", 5: "Eczema Crónico", 6: "Pitiriasis Rubra Pilaris"
}

print(f"\n✅ Predicción: {class_mapping.get(pred, 'Desconocido')}")
print(f"✅ Confianza: {conf}%")
