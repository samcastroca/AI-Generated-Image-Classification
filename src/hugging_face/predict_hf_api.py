#!/usr/bin/env python3
import argparse
import os
import sys
import io
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(128, 128)):
    """Resize -> RGB -> normalize to [0,1] -> add batch dim (kept for diagnostics)."""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_resized = image.resize(target_size)
    arr = np.array(image_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr, image  # return preprocessed array and original PIL image

def call_hf_inference(api_url, hf_token, image_path, timeout=30):
    """
    Call HF Inference API by sending raw image bytes.
    Expected responses usually: [{'label': 'REAL', 'score': 0.9}, {...}]
    """
    headers = {"Authorization": f"Bearer {hf_token}"}
    # read bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    try:
        # Most HF vision models accept raw image bytes POSTed to the model endpoint
        resp = requests.post(api_url, headers=headers, data=image_bytes, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Hugging Face API request failed: {e}\nResponse: {getattr(e, 'response', None)}")

    try:
        result = resp.json()
    except ValueError:
        raise RuntimeError(f"Invalid JSON response from HF API: {resp.text[:500]}")

    return result

def parse_hf_result(result):
    """
    Normalize HF response into a list of dicts like: [{'label': 'REAL', 'score': 0.9}, ...]
    Then decide final class (REAL/FAKE) and top score.
    """
    # If result is a dict with 'error'
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(f"Hugging Face inference error: {result['error']}")

    # If the response is something else, attempt to normalize:
    normalized = None

    if isinstance(result, list) and all(isinstance(x, dict) and 'label' in x for x in result):
        normalized = result
    elif isinstance(result, dict):
        # Some endpoints return {"label": "...", "score": ...} directly
        if 'label' in result and 'score' in result:
            normalized = [result]
        # Some return {"scores": [...], "labels": [...]}
        elif 'labels' in result and 'scores' in result:
            labels = result['labels']
            scores = result['scores']
            normalized = [{'label': str(l), 'score': float(s)} for l, s in zip(labels, scores)]
        elif 'scores' in result and isinstance(result['scores'], list):
            # fall back to numeric indices as labels
            scores = result['scores']
            normalized = [{'label': str(i), 'score': float(s)} for i, s in enumerate(scores)]
        else:
            # Unknown shape - try to coerce if it's dict of scalar probabilities
            try:
                normalized = [{'label': k, 'score': float(v)} for k, v in result.items()]
            except Exception:
                raise RuntimeError(f"Unrecognized HF response format: {result}")
    else:
        raise RuntimeError(f"Unrecognized HF response type: {type(result)} - content: {result}")

    # Sort by score descending
    normalized = sorted(normalized, key=lambda x: x.get('score', 0.0), reverse=True)

    top = normalized[0]
    top_label = str(top.get('label')).upper()
    top_score = float(top.get('score', 0.0))

    # Decide mapping to REAL/FAKE
    # Heuristics:
    # - If label string contains REAL/FAKE -> use it
    # - If label is numeric '0' or '1' -> map 1 -> REAL, 0 -> FAKE
    # - Otherwise, if labels are like 'LABEL_0' / 'LABEL_1' try to infer
    final = None
    if "REAL" in top_label:
        final = ("REAL", top_score)
    elif "FAKE" in top_label or "DEEPFAKE" in top_label or "SYNTH" in top_label:
        final = ("FAKE", top_score)
    else:
        # try numeric
        try:
            num = int(top_label.strip().split()[-1]) if top_label.strip().isdigit() else int(top_label)
            if num == 1:
                final = ("REAL", top_score)
            elif num == 0:
                final = ("FAKE", top_score)
        except Exception:
            # fallback: if the second label exists, assume it's the complement, use score threshold
            # score near 1 => top_label is correct; we will fallback to mapping based on presence of '1'/'0' in labels
            # as ultimate fallback: decide REAL if top_score > 0.5 and label isn't obviously negative
            final = ("REAL", top_score) if top_score > 0.5 else ("FAKE", 1 - top_score)

    return normalized, final

def visualize_and_save(original_pil_image, preprocessed_arr, normalized_preds, final_pred, image_path, show_plot=True):
    """
    Display original image + resized normalized array and save figure to disk.
    """
    # preprocessed_arr has shape (1, H, W, C)
    display_array = preprocessed_arr[0]  # (H, W, C) normalized to [0,1]
    # Ensure display_array is in [0,1]
    display_array = np.clip(display_array, 0.0, 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.imshow(original_pil_image)
    ax1.set_title("Original Input Image")
    ax1.axis("off")

    ax2.imshow(display_array)
    ax2.set_title("Resized Image (model input)")
    ax2.axis("off")

    # Compose subtitle showing top predictions
    preds_text = "\n".join([f"{p['label']}: {p['score']:.3f}" for p in normalized_preds[:3]])
    fig.suptitle(f"Final Prediction: {final_pred[0]} (score: {final_pred[1]:.3f})\nTop preds:\n{preds_text}", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = os.path.splitext(image_path)[0] + "_hf_results.png"
    plt.savefig(output_path)
    print(f"Results visualization saved to: {output_path}")

    if show_plot:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Inference using Hugging Face Inference API (REAL vs FAKE classifier).")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--api_url", type=str, default="https://api-inference.huggingface.co/models/juandaram/deepfake-detector",
                        help="Hugging Face model API URL (default to your model)")
    parser.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=(128, 128),
                        help="Target size (h w) used to preprocess image (default 128 128)")
    parser.add_argument("--quiet", action="store_true", help="Only print FINAL label (REAL or FAKE) and exit")
    parser.add_argument("--no-plot", dest="show_plot", action="store_false", help="Do not show matplotlib plot (still saves file)")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP request timeout in seconds")
    args = parser.parse_args()

    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Image file not found: {args.image_path}", file=sys.stderr)
        sys.exit(2)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set.", file=sys.stderr)
        print("Export it as: export HF_TOKEN='hf_...' (or set it in your environment).", file=sys.stderr)
        sys.exit(2)

    # Preprocess (kept for visualization & to confirm shape/range)
    preprocessed_arr, original_pil = preprocess_image(args.image_path, target_size=tuple(args.size))
    print(f"Preprocessed image shape: {preprocessed_arr.shape}, range: [{preprocessed_arr.min():.4f}, {preprocessed_arr.max():.4f}]")

    # Call HF
    print("Calling Hugging Face Inference API...")
    raw_result = call_hf_inference(args.api_url, hf_token, args.image_path, timeout=args.timeout)
    normalized_preds, final = parse_hf_result(raw_result)

    # final is tuple ("REAL"/"FAKE", score)
    final_label, final_score = final

    # If quiet, print only the label (single word) and exit
    if args.quiet:
        print(final_label)
        return

    # Print debug/readable info
    print("\n--- Inference result (normalized) ---")
    for p in normalized_preds:
        print(f"{p['label']}: {p['score']:.6f}")
    print(f"\nFinal decision: {final_label} (confidence: {final_score:.6f})")

    # Visualize and save
    visualize_and_save(original_pil, preprocessed_arr, normalized_preds, final, args.image_path, show_plot=args.show_plot)

if __name__ == "__main__":
    main()
