# 🌐 DIAGNÓSTICO DERMATOLÓGICO INTELIGENTE  
### 🩺 Sistema híbrido con Visión Computacional + Random Forest  
**Universidad Privada Antenor Orrego – Facultad de Ingeniería**

![Status](https://img.shields.io/badge/estado-en%20desarrollo-blue)
![Python](https://img.shields.io/badge/Python-3.9+-yellow)
![Flask](https://img.shields.io/badge/Flask-API%20REST-black)
![Model](https://img.shields.io/badge/IA-Random%20Forest-green)
![License](https://img.shields.io/badge/licencia-académica-purple)

---

## 📌 1. Descripción del Proyecto
Este proyecto implementa un **sistema experto de apoyo a la decisión clínica** para el diagnóstico de enfermedades dermatológicas eritroescamosas.  

El sistema analiza una imagen dermatológica, extrae características y determina la probabilidad de que pertenezca a una de las siguientes **6 patologías**:

- 🧬 **Psoriasis**  
- 🧪 **Dermatitis Seborreica**  
- 🌿 **Liquen Plano**  
- 🌸 **Pitiriasis Rosada**  
- 🧼 **Eczema Crónico**  
- 🔶 **Pitiriasis Rubra Pilaris**

El modelo utiliza **Visión Computacional (Pillow)** y un algoritmo **Random Forest** entrenado con datos clínicos reales.

---

## 🏗️ 2. Arquitectura del Sistema

El sistema opera sobre una arquitectura **Serverless basada en microservicios**, lo que garantiza:

- Alta disponibilidad  
- Escalabilidad automática  
- Despliegue rápido en producción  

### 🔧 Componentes Principales

#### **Frontend (/public)**
- Interfaz web desarrollada con **HTML5 + Tailwind CSS**
- Permite subir imágenes, ejecutar análisis y visualizar resultados.

#### **Backend (/api)**
- API REST construida con **Flask (Python)**  
- Procesamiento de imágenes con **Pillow**  
- Ejecución del modelo entrenado con **Scikit-learn**

---

## 📁 3. Estructura del Proyecto

## 📁 Estructura del Proyecto

```bash
📦 proyecto_upao
├── 📂 api
│   └── 🐍 index.py              # Endpoints (/api/predict, /api/analyze_image)
│
├── 📂 public
│   └── 🌐 index.html            # Interfaz principal del sistema
│
├── 🤖 modelo_derma.pkl         # Modelo Random Forest entrenado
├── 📄 requirements.txt         # Dependencias de Python
└── ⚙️ vercel.json              # Configuración de despliegue en Vercel

---

## ⚙️ 4. Instalación y Ejecución Local

### ✔ Prerrequisitos
- Python **3.9+**  
- Git  

### ✔ Instalación

#### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/proyecto-derma-upao.git
cd proyecto-derma-upao
