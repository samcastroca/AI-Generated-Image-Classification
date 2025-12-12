import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import os

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess the image to match the training data generator:
    Only rescale to [0, 1] - the model already has preprocess_input baked in
    """
    # Load the image
    image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image
    image = image.resize(target_size)
    
    # Convert to array and normalize to [0, 1] (matching ImageDataGenerator rescale=1./255)
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def load_tflite_model(model_path):
    """
    Load the TensorFlow Lite model
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    return interpreter

def predict_with_tflite(interpreter, input_data):
    """
    Perform prediction using the TFLite interpreter
    """
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data

def visualize_results(original_image_path, predictions, args=None):
    """
    Create a visualization of the results showing both original and preprocessed images
    """
    # Load and display the original image
    original_image = Image.open(original_image_path)
    
    # For display purposes, show the image normalized to [0,1]
    display_image = original_image.resize((128, 128))
    display_array = np.array(display_image) / 255.0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display the original image
    ax1.imshow(original_image)
    ax1.set_title('Original Input Image')
    ax1.axis('off')
    
    # Display the resized image
    ax2.imshow(display_array)
    ax2.set_title('Resized Image (128x128)')
    ax2.axis('off')
    
    # Determine the final prediction based on the model output
    # During training: {'FAKE': 0, 'REAL': 1}
    # Sigmoid output: values close to 0 = FAKE, values close to 1 = REAL
    output_value = predictions[0][0]
    
    if output_value > 0.5:
        predicted_class = 1  # REAL
        confidence = output_value
        final_prediction = "REAL"
    else:
        predicted_class = 0  # FAKE
        confidence = 1 - output_value
        final_prediction = "FAKE"
    
    fig.suptitle(f'Final Prediction: {final_prediction} (Confidence: {confidence:.3f})', fontsize=16)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.splitext(original_image_path)[0] + '_results.png'
    plt.savefig(output_path)
    print(f"Results visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Local inference script for image classification')
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
    
    print(f"Loading model from: {args.model_path}")
    interpreter = load_tflite_model(args.model_path)
    
    # Get the model's input details to determine the correct input size
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    # Extract height and width from the input shape
    height, width = input_shape[1], input_shape[2]
    print(f"Model expects input size: {width}x{height}")
    
    print(f"Preprocessing image: {args.image_path}")
    input_data = preprocess_image(args.image_path, target_size=(height, width))
    
    print("Running inference...")
    predictions = predict_with_tflite(interpreter, input_data)
    
    print("Generating visualization...")
    visualize_results(args.image_path, predictions)
    
    # Determine the final prediction
    # Training classes: {'FAKE': 0, 'REAL': 1}
    # Sigmoid output > 0.5 means class 1 (REAL)
    output_value = predictions[0][0]
    
    if output_value > 0.5:
        predicted_class = 1  # REAL
        confidence = output_value
        final_prediction = "REAL"
    else:
        predicted_class = 0  # FAKE
        confidence = 1 - output_value
        final_prediction = "FAKE"
    
    print(f"\nRaw model output: {output_value:.6f}")
    print(f"Predicted class (0=FAKE, 1=REAL): {predicted_class}")
    print(f"Final Prediction: {final_prediction}")
    print(f"Confidence: {confidence:.3f}")
    
    # Debug information
    print(f"\n--- Debug Info ---")
    print(f"Input data shape: {input_data.shape}")
    print(f"Input data range: [{input_data.min():.4f}, {input_data.max():.4f}]")
    print(f"Input data mean: {input_data.mean():.4f}")

if __name__ == "__main__":
    main()