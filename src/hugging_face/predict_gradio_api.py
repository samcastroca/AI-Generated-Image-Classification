"""
Script para consumir la API de Gradio Space en Hugging Face
NO descarga el modelo - solo envía la imagen
"""
import argparse
import requests
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# URL de tu Gradio Space (cambiar después de crear el Space)
GRADIO_API_URL = "https://juandaram-deepfake-detector-api.hf.space/api/predict"

def predict_image(image_path):
    """
    Envía la imagen a la API de Gradio y obtiene la predicción
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    
    print(f"Analizando imagen: {image_path}")
    print(f"Usando API: {GRADIO_API_URL}")
    
    # Abrir imagen
    with open(image_path, 'rb') as f:
        files = {'data': f}
        
        try:
            # Enviar a Gradio API
            response = requests.post(GRADIO_API_URL, files=files, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Formato de respuesta de Gradio:
            # {"data": [{"label": "REAL", "confidences": [{"label": "FAKE", "confidence": 0.1}, {"label": "REAL", "confidence": 0.9}]}]}
            
            if 'data' in result:
                confidences = result['data'][0].get('confidences', [])
                
                # Encontrar REAL y FAKE
                real_conf = next((c['confidence'] for c in confidences if c['label'] == 'REAL'), 0.5)
                fake_conf = next((c['confidence'] for c in confidences if c['label'] == 'FAKE'), 0.5)
                
                # Determinar predicción
                if real_conf > fake_conf:
                    prediction = "REAL"
                    confidence = real_conf
                else:
                    prediction = "FAKE"
                    confidence = fake_conf
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'real_score': real_conf,
                    'fake_score': fake_conf
                }
            else:
                raise Exception(f"Formato de respuesta inesperado: {result}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error de red: {e}")

def visualize_results(image_path, result):
    """
    Visualiza los resultados
    """
    # Cargar imagen
    image = Image.open(image_path)
    
    # Redimensionar para display
    display_image = image.resize((128, 128))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original
    ax1.imshow(image)
    ax1.set_title('Original Input Image')
    ax1.axis('off')
    
    # Resized
    ax2.imshow(display_image)
    ax2.set_title('Resized Image (128x128)')
    ax2.axis('off')
    
    # Título con resultado
    prediction = result['prediction']
    confidence = result['confidence']
    
    fig.suptitle(f'Final Prediction: {prediction} (Confidence: {confidence:.3f})', fontsize=16)
    
    plt.tight_layout()
    
    # Guardar
    output_path = os.path.splitext(image_path)[0] + '_results.png'
    plt.savefig(output_path)
    print(f"Results visualization saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict using Gradio Space API')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--api_url', type=str, default=GRADIO_API_URL,
                        help='Gradio API URL')
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not show plot')
    
    args = parser.parse_args()
    
    try:
        # Actualizar URL si se proporciona
        global GRADIO_API_URL
        GRADIO_API_URL = args.api_url
        
        # Predecir
        result = predict_image(args.image_path)
        
        # Imprimir resultados
        print("\n" + "="*60)
        print("RESULTADO DEL ANÁLISIS")
        print("="*60)
        print(f"Imagen: {args.image_path}")
        print(f"Predicción: {result['prediction']}")
        print(f"Confianza: {result['confidence']:.3f}")
        print(f"\nScores:")
        print(f"  REAL: {result['real_score']:.3f}")
        print(f"  FAKE: {result['fake_score']:.3f}")
        print("="*60)
        
        # Visualizar
        if not args.no_plot:
            visualize_results(args.image_path, result)
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()