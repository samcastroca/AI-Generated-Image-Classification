# Local Inference Script for Image Classification

This script performs local inference on an image using a TensorFlow Lite model and generates a visualization of the results.

## Features

- Takes an image path as input
- Loads and runs a TensorFlow Lite model
- Generates a graph with prediction results
- Creates a visualization combining the input image and prediction probabilities

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib
- Pillow (PIL)

Install the required packages using pip:

```bash
pip install tensorflow numpy matplotlib pillow
```

## Usage

Run the script with the following command:

```bash
python src/inference_script.py <image_path> [options]
```

### Arguments

- `image_path`: Path to the input image (required)

### Options

- `--model_path`: Path to the TFLite model (default: `models/model_litert.tflite`)
- `--class_names`: Class names for the prediction (optional)

### Examples

Basic usage:
```bash
python src/local_predict/inference_script.py my_image.jpg
```

Specify a custom model path:
```bash
python src/local_predict/inference_script.py --model_path /path/to/my/model.tflite
```

Provide class names for the prediction:
```bash
python src/inference_script.py my_image.jpg --class_names cat dog bird
```

## Output

The script will:

1. Load the model and input image
2. Preprocess the image to match the model's expected input size
3. Run inference using the model
4. Generate a visualization showing:
   - The input image on the left
   - A bar chart of prediction probabilities on the right
5. Save the visualization as `<input_image_name>_results.png`
6. Print the prediction results to the console

## Troubleshooting

- If you get a dimension mismatch error, make sure your input image matches the model's expected input size
- Ensure that the model file exists at the specified path
- If you get a module import error, make sure all required packages are installed

## Note

This script uses TensorFlow Lite for optimized inference, which is especially useful for running models on mobile or embedded devices.
