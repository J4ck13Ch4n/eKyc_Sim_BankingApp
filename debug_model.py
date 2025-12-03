import warnings
# Suppress the specific numpy/importlib warning on Windows
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert('RGB')

def preprocess_image(image):
    # Resize + Normalize ImageNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image)

def print_tensor_stats(name, tensor):
    print(f"[DEBUG] {name} mean={tensor.mean().item():.5f} std={tensor.std().item():.5f} min={tensor.min().item():.5f} max={tensor.max().item():.5f}")

def main():
    parser = argparse.ArgumentParser(description='Debug eKYC Model')
    parser.add_argument('--id', type=str, required=True, help='Path to ID card image')
    parser.add_argument('--face', type=str, required=True, help='Path to Face image')
    parser.add_argument('--model', type=str, default='android/app/src/main/assets/ekyc_model_mobile.ptl', help='Path to model file')
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    try:
        model = torch.jit.load(args.model)
        model.eval()
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Preprocessing images...")
    try:
        id_img = load_image(args.id)
        face_img = load_image(args.face)

        # ID tensor: [1, 3, 224, 224]
        id_tensor = preprocess_image(id_img).unsqueeze(0)

        # Face video: repeat frame 8 times
        face_tensor_single = preprocess_image(face_img)
        face_tensor = face_tensor_single.unsqueeze(0).repeat(8, 1, 1, 1).unsqueeze(0)

        print(f"ID Tensor Shape: {id_tensor.shape}")
        print(f"Face Tensor Shape: {face_tensor.shape}")

        # DEBUG VALUE DISTRIBUTION
        print_tensor_stats("ID Tensor", id_tensor)
        print_tensor_stats("Face Tensor", face_tensor)

    except Exception as e:
        print(f"Error processing images: {e}")
        return

    print("Running inference...")
    with torch.no_grad():
        try:
            outputs = model(id_tensor, face_tensor)

            # DEBUG raw output type and values
            print(f"[DEBUG] Model output types: {type(outputs)}")
            if isinstance(outputs, tuple):
                print(f"[DEBUG] Output tuple length: {len(outputs)}")
            else:
                print("[ERROR] Model did NOT return tuple!")

            print("[DEBUG] Raw output tensors:")
            print(outputs)

            liveness_score = outputs[0].item()
            matching_score = outputs[1].item()

            print("-" * 40)
            print(f"Liveness Score: {liveness_score:.6f}")
            print(f"Matching Score: {matching_score:.6f}")
            print("-" * 40)

        except Exception as e:
            print(f"Inference failed: {e}")
            return

if __name__ == "__main__":
    main()
