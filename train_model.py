"""
Script para entrenar modelo Random Forest con dataset UCI Dermatology
"""
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

print("📥 Descargando dataset de UCI...")
dermatology = fetch_ucirepo(id=33) 

X = dermatology.data.features 
y = dermatology.data.targets

print(f"✅ Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
print(f"📊 Clases: {y['class'].unique()}")

# Manejo de valores faltantes
print("🔧 Procesando valores faltantes...")
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
# Convertir a numpy array para evitar problemas con nombres de características
X_imputed = imputer.fit_transform(X.values)

# Split datos
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y.values.ravel(), test_size=0.2, random_state=42, stratify=y
)

print("🌲 Entrenando Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Modelo entrenado con éxito!")
print(f"📈 Precisión en test: {accuracy * 100:.2f}%")
print("\n📊 Reporte detallado:")
print(classification_report(y_test, y_pred, 
    target_names=['Psoriasis', 'Dermatitis Seborreica', 'Liquen Plano', 
                  'Pitiriasis Rosada', 'Eczema Crónico', 'Pitiriasis Rubra Pilaris']))

# Guardar modelo
print("💾 Guardando modelo...")
joblib.dump(model, 'modelo_derma.pkl')
joblib.dump(imputer, 'imputer.pkl')

print("✅ Modelo y preprocesador guardados exitosamente!")
print("📁 Archivos: modelo_derma.pkl, imputer.pkl")
