import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess the image to the required input size for the model
    """
    # Load the image
    image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def debug_inference(image_path, model_path):
    """
    Perform detailed debugging of the inference process
    """
    print(f"Loading model from: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    print(f"Preprocessing image: {image_path}")
    input_data = preprocess_image(image_path)
    print(f"Input shape: {input_data.shape}")
    print(f"Input min/max: {input_data.min():.3f}/{input_data.max():.3f}")
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Raw output: {output_data}")
    print(f"Output shape: {output_data.shape}")
    print(f"Output value: {output_data[0][0]:.6f}")
    
    # Apply sigmoid if needed (though for a trained model with sigmoid output layer, 
    # the TFLite model typically already applies the activation)
    raw_output = output_data[0][0]
    
    # For a binary classifier with sigmoid output:
    # Value closer to 0 means REAL, value closer to 1 means FAKE
    if raw_output > 0.5:
        prediction = "FAKE"
        confidence = raw_output
    else:
        prediction = "REAL"
        confidence = 1 - raw_output
        
    print(f"Predicted class: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show the image
    original_image = Image.open(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    plt.title(f'Image: {prediction} (Confidence: {confidence:.3f})\nRaw output: {raw_output:.6f}')
    plt.axis('off')
    plt.show()
    
    return prediction, confidence

def main():
    parser = argparse.ArgumentParser(description='Debug inference script for image classification')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='models/model_litert.tflite',
                        help='Path to the TFLite model (default: models/model_litert.tflite)')
    
    args = parser.parse_args()
    
    # Validate input image path
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
    # Validate model path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    debug_inference(args.image_path, args.model_path)

if __name__ == "__main__":
    main()
