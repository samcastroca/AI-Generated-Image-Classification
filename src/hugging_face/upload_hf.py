"""
Script para subir el modelo de detección REAL vs FAKE a Hugging Face
"""
from huggingface_hub import HfApi, create_repo
import os
import shutil

def upload_model_to_huggingface(
    model_path="models/saved_model",
    tflite_path="models/model_litert.tflite",
    repo_name="deepfake-detector",
    username=None  # Tu username de Hugging Face
):
    """
    Sube el modelo a Hugging Face Hub
    
    Args:
        model_path: Ruta al SavedModel
        tflite_path: Ruta al modelo TFLite
        repo_name: Nombre del repositorio (ej: "deepfake-detector")
        username: Tu username de Hugging Face
    """
    
    if username is None:
        username = input("Ingresa tu username de Hugging Face: ")
    
    # Nombre completo del repo
    repo_id = f"{username}/{repo_name}"
    
    print(f"Creando repositorio: {repo_id}")
    
    # Crear API
    api = HfApi()
    
    # Crear repositorio (si no existe)
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"✓ Repositorio creado/verificado: {repo_id}")
    except Exception as e:
        print(f"Error creando repositorio: {e}")
        return
    
    # Crear README.md
    readme_content = f"""---
license: mit
tags:
- image-classification
- deepfake-detection
- tensorflow
- tflite
datasets:
- custom
metrics:
- accuracy
---

# Deepfake Detector Model

Este modelo detecta si una imagen es REAL o FAKE (generada/manipulada).

## Modelo

- **Arquitectura:** ResNet50 con Transfer Learning
- **Framework:** TensorFlow / TensorFlow Lite
- **Input:** Imágenes RGB de 128x128 píxeles
- **Output:** Probabilidad sigmoid (0=FAKE, 1=REAL)
- **Threshold:** 0.5

## Uso

### Con TensorFlow Lite (Python)

```python
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Descargar modelo
model_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="model.tflite"
)

# Cargar modelo
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Preparar imagen
image = Image.open("tu_imagen.jpg").convert('RGB')
image = image.resize((128, 128))
img_array = np.array(image, dtype=np.float32) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# Predecir
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], img_batch)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

prediction = "REAL" if output[0][0] > 0.5 else "FAKE"
confidence = output[0][0] if output[0][0] > 0.5 else (1 - output[0][0])

print(f"Prediction: {{prediction}}")
print(f"Confidence: {{confidence:.3f}}")
```

### Con SavedModel (Python)

```python
import tensorflow as tf
from huggingface_hub import snapshot_download

# Descargar modelo completo
model_dir = snapshot_download(repo_id="{repo_id}")

# Cargar modelo
model = tf.saved_model.load(f"{{model_dir}}/saved_model")
infer = model.signatures['serving_default']

# Usar igual que arriba...
```

## Métricas

- Validation Accuracy: ~84%
- Training Epochs: 5

## Clases

- 0: FAKE (imagen generada/manipulada)
- 1: REAL (imagen auténtica)

## Preprocesamiento

Las imágenes deben:
1. Convertirse a RGB
2. Redimensionarse a 128x128
3. Normalizarse dividiendo por 255.0 (rango [0, 1])

## Limitaciones

- El modelo puede tener sesgo hacia la clase FAKE
- Funciona mejor con imágenes similares al dataset de entrenamiento
- Requiere imágenes de buena calidad

## Licencia

MIT

## Contacto

Para preguntas o problemas, abre un issue en el repositorio.
"""
    
    # Guardar README
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✓ README.md creado")
    
    # Subir archivos
    files_to_upload = []
    
    # 1. README
    files_to_upload.append(("README.md", "README.md"))
    
    # 2. TFLite model
    if os.path.exists(tflite_path):
        files_to_upload.append((tflite_path, "model.tflite"))
        print(f"✓ TFLite model encontrado: {tflite_path}")
    else:
        print(f"⚠️  TFLite model no encontrado: {tflite_path}")
    
    # 3. SavedModel (comprimido)
    if os.path.exists(model_path):
        # Comprimir SavedModel
        print("Comprimiendo SavedModel...")
        shutil.make_archive("saved_model", 'zip', model_path)
        files_to_upload.append(("saved_model.zip", "saved_model.zip"))
        print("✓ SavedModel comprimido")
    else:
        print(f"⚠️  SavedModel no encontrado: {model_path}")
    
    # Subir archivos
    print(f"\nSubiendo archivos a {repo_id}...")
    for local_path, remote_path in files_to_upload:
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"✓ Subido: {remote_path}")
        except Exception as e:
            print(f"✗ Error subiendo {remote_path}: {e}")
    
    # Limpiar archivos temporales
    if os.path.exists("README.md"):
        os.remove("README.md")
    if os.path.exists("saved_model.zip"):
        os.remove("saved_model.zip")
    
    print(f"\n🎉 ¡Modelo subido exitosamente!")
    print(f"🔗 Ver en: https://huggingface.co/{repo_id}")
    
    return repo_id


if __name__ == "__main__":
    import sys
    
    # Puedes pasar argumentos por línea de comandos
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = None
    
    if len(sys.argv) > 2:
        repo_name = sys.argv[2]
    else:
        repo_name = "deepfake-detector"
    
    repo_id = upload_model_to_huggingface(
        model_path="models/saved_model",
        tflite_path="models/model_litert.tflite",
        repo_name=repo_name,
        username=username
    )