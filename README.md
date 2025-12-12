# 🔍 Detección de Imágenes: Real vs. AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samcastroca/proyecto-pdi/blob/main/notebooks/train.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/juandaram/deepfake-detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Clasificación Binaria con Transfer Learning (ResNet50)**

> Proyecto de Deep Learning para detectar imágenes generadas por IA vs. imágenes reales.

**Autores:** Samuel Castro, Juan David Ramírez Ortiz

---

## 📋 Tabla de Contenidos

- [El Problema](#-el-problema)
- [Solución Propuesta](#-solución-propuesta)
- [Dataset](#-dataset)
- [Arquitectura del Modelo](#-arquitectura-del-modelo)
- [Resultados](#-resultados)
- [Instalación](#-instalación)
- [Uso](#-uso)
  - [Inferencia Local](#inferencia-local)
  - [API de Hugging Face](#api-de-hugging-face)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Limitaciones](#-limitaciones)
- [Conclusiones](#-conclusiones)
- [Licencia](#-licencia)

---

## 🎯 El Problema

### El Desafío: Deepfakes y Generación AI

La proliferación de modelos generativos permite crear imágenes sintéticas indistinguibles a simple vista. Es necesario automatizar la distinción entre contenido auténtico y generado.

**Impacto:**
- 📰 Prevención de desinformación (Fake News)
- 🔐 Validación de identidad y seguridad digital
- 🌐 Filtrado de contenido en redes sociales

---

## 💡 Solución Propuesta

| Aspecto | Descripción |
|---------|-------------|
| **Tarea** | Clasificación Binaria de Imágenes |
| **Clases** | `REAL` (1) vs. `FAKE` (0) |
| **Modelo** | Red Neuronal Convolucional (CNN) |
| **Técnica** | Transfer Learning (Fine-tuning parcial) |
| **Base** | ResNet50 pre-entrenada en ImageNet |

---

## 📊 Dataset

El modelo fue entrenado utilizando el dataset **CIFAKE**, que contiene imágenes reales e imágenes generadas por IA.

### Estructura del Dato

| Característica | Valor |
|----------------|-------|
| **Resolución Original** | 32 × 32 píxeles (RGB) |
| **Resolución de Entrada** | 128 × 128 píxeles (upscaling) |
| **Normalización** | Rescale 1./255 |

### Distribución

| Set | Cantidad | Distribución |
|-----|----------|--------------|
| Entrenamiento (Train) | 100,000 imágenes | Balanceado (50/50) |
| Prueba (Test) | 20,000 imágenes | Balanceado (50/50) |
| **Total** | **120,000 imágenes** | |

---

## 🏗️ Arquitectura del Modelo

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────┐
│   Input     │───▶│    ResNet50      │───▶│   Últimas 10    │───▶│ Global Avg  │───▶│  Dense 256  │───▶│ Sigmoid │
│ 128×128×3   │    │   (Congelado)    │    │    Capas        │    │  Pooling    │    │ + Dropout   │    │  Output │
└─────────────┘    │   (ImageNet)     │    │  (Trainable)    │    └─────────────┘    │   (0.5)     │    └─────────┘
                   └──────────────────┘    └─────────────────┘                       └─────────────┘
```

### Configuración del Modelo

- **Base:** ResNet50 (Weights: ImageNet)
- **Congelamiento:** Todas las capas excepto las últimas 10
- **Cabezal (Head):**
  - GlobalAveragePooling2D
  - Dense (256 neuronas, ReLU) + Dropout (0.5)
  - Salida: Dense (1 neurona, Sigmoid)
- **Optimizador:** Adam (lr=1e-5)
- **Loss:** Binary Crossentropy

### Componentes

| Componente | Descripción |
|------------|-------------|
| Extractor de características | Capas congeladas de ResNet50 |
| Fine-tuning | Últimas 10 capas entrenables |
| Clasificador | Capas densas personalizadas |

---

## 📈 Resultados

### Métricas Finales (Época 5)

| Métrica | Valor |
|---------|-------|
| **Train Accuracy** | 87.25% |
| **Val Accuracy** | 84.90% (Pico: 86.76%) |
| **Train Loss** | 0.3026 |

### Optimización del Modelo

Comparativa de formatos para despliegue:

| Formato | Peso Aprox. | Velocidad | Caso de Uso |
|---------|-------------|-----------|-------------|
| Keras (.h5/.keras) | ~95 MB | Lento | Entrenamiento |
| TorchScript / ONNX | ~90 MB | Medio | Servidor Cloud |
| **LiteRT (TFLite)** | **~25 MB** | **Rápido** | **Móvil / Edge** |

---

## ⚙️ Instalación

### Requisitos Previos

- Python 3.7+
- pip

### Clonar el Repositorio

```bash
git clone https://github.com/samcastroca/proyecto-pdi.git
cd proyecto-pdi
```

### Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Dependencias Principales

```bash
pip install tensorflow numpy matplotlib pillow requests
```

---

## 🚀 Uso

### Inferencia Local

El script de inferencia local utiliza el modelo TFLite optimizado para máxima velocidad.

#### Flujo de Ejecución

1. **Carga:** Recibe la ruta de la imagen y del modelo `.tflite` por línea de comandos
2. **Preprocesamiento:**
   - Redimensiona a 128 × 128 píxeles
   - Normaliza píxeles al rango [0, 1]
   - Añade dimensión Batch (1, 128, 128, 3)
3. **Inferencia:** Usa `tf.lite.Interpreter` (sin cargar Keras completo)
4. **Post-procesamiento:** Decodifica la salida Sigmoid y genera visualización

#### Comando Básico

```bash
python src/local_predict/inference_script.py <ruta_imagen>
```

#### Ejemplos

```bash
# Usar modelo por defecto
python src/local_predict/inference_script.py imgs/fake/5.jpg

# Especificar modelo personalizado
python src/local_predict/inference_script.py mi_imagen.jpg --model_path models/model.tflite
```

#### Argumentos

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `image_path` | Ruta a la imagen de entrada (requerido) | - |
| `--model_path` | Ruta al modelo TFLite | `models/model_litert.tflite` |

#### Salida

El script genera:
- Visualización con la imagen original y redimensionada
- Predicción final: `REAL` o `FAKE`
- Nivel de confianza (0-1)
- Archivo de imagen con resultados: `<nombre_imagen>_results.png`

#### Lógica de Clasificación

```python
# Preprocesamiento (idéntico al entrenamiento)
image = image.resize((128, 128))
input_data = np.array(image) / 255.0

# Inferencia TFLite
interpreter.allocate_tensors()
interpreter.set_tensor(idx, input_data)
interpreter.invoke()
output = interpreter.get_tensor(idx)

# Clasificación binaria (Umbral: 0.5)
if output[0][0] > 0.5:
    label = "REAL"   # Confianza = output
else:
    label = "FAKE"   # Confianza = 1 - output
```

---

### API de Hugging Face

El modelo está desplegado en Hugging Face Spaces con una interfaz Gradio para pruebas rápidas.

#### Demo Web

🌐 **URL:** [[https://juandaram-deepfake-detector-api.hf.space](https://juandaram-deepfake-detector-api.hf.space])

Simplemente:
1. Sube una imagen
2. Obtén la probabilidad Real/Fake

#### Usar desde Python

```bash
python src/hugging_face/predict_gradio_api.py <ruta_imagen>
```

---

## 📁 Estructura del Proyecto

```
proyecto-pdi/
├── docs/                          # Documentación
├── imgs/                          # Imágenes de ejemplo
│   ├── fake/                      # Imágenes fake para testing
│   └── real/                      # Imágenes reales para testing
├── models/                        # Modelos entrenados
│   ├── model.tflite              # Modelo TFLite optimizado
│   └── saved_model/              # Modelo SavedModel de TensorFlow
├── notebooks/
│   ├── train.ipynb               # Notebook de entrenamiento
│   └── convert_litert.ipynb      # Conversión a TFLite
├── src/
│   ├── local_predict/
│   │   ├── inference_script.py   # Script de inferencia local
│   │   └── debug_inference.py    # Script de debugging
│   ├── hugging_face/
│   │   ├── predict_gradio_api.py # Cliente API de Gradio
│   │   ├── predict_hf_api.py     # Cliente API de HF
│   │   └── upload_hf.py          # Script para subir a HF
│   ├── data/                     # Scripts de procesamiento de datos
│   ├── features/                 # Feature engineering
│   ├── models/                   # Definiciones de modelos
│   └── visualization/            # Scripts de visualización
├── reports/
│   └── figures/                  # Gráficas y figuras generadas
├── references/                   # Recursos y referencias
├── requirements.txt              # Dependencias del proyecto
├── setup.py                      # Configuración del paquete
├── Makefile                      # Comandos de automatización
└── README.md                     # Este archivo
```

---

## ⚠️ Limitaciones

| Limitación | Descripción |
|------------|-------------|
| **Resolución** | El upscale de 32 a 128px puede introducir artefactos no deseados |
| **Épocas** | Se entrenó solo por 5 épocas; podría mejorar con más tiempo |
| **Generalización** | Necesario probar con imágenes de alta calidad (no CIFAR/thumbnails) |

---

## 📝 Conclusiones

- **Eficacia de ResNet50:** A pesar de usar imágenes pequeñas escaladas, el modelo pre-entrenado logra casi un 87% de precisión rápidamente.

- **Fine-Tuning:** Descongelar las últimas 10 capas fue crucial para adaptar las características de ImageNet al dominio sintético.

- **Escalabilidad:** Con 120,000 imágenes, el dataset es robusto, pero el modelo se beneficiaría de Data Augmentation más agresivo para reducir el overfitting leve.

---

## 🔧 Desarrollo

### Entrenar el Modelo

Abre el notebook en Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samcastroca/proyecto-pdi/blob/main/notebooks/train.ipynb)

### Convertir a TFLite

```bash
# Usar el notebook de conversión
jupyter notebook notebooks/convert_litert.ipynb
```

### Tests

```bash
python src/test_model.py
python src/test_savedmodel.py
```

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 👥 Autores

- **Samuel Castro** - [@samcastroca](https://github.com/samcastroca)
- **Juan David Ramírez Ortiz** [@juandaram](https://github.com/juandaram)

---

<p align="center">
  <i>Proyecto de Deep Learning - 2025</i>
</p>
