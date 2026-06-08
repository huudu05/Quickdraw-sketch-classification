import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from src.config import CLASSES
from src.model import QuickDrawCNN
from src.utils import preprocess_input

def get_args():
    parser = argparse.ArgumentParser(description="QuickDraw Sketch Classification Inference")
    parser.add_argument(
        "--model_path", type=str, default="checkpoints/best_model.pth", help="Path to the trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--class_name", type=str, default="apple", help="Target class from which a test sample will be selected"
    )
    parser.add_argument(
        "--sample_idx", type=int, default=0, help="Index of the sample within the selected class dataset"
    )
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}. Please train the model first by running train.py.")
        return


    model = QuickDrawCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()


    raw_data_path = f"data/{args.class_name}.npy"
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file not found at {raw_data_path}. Unable to extract the test image.")
        return

    print(f"Extracting sample #{args.sample_idx} from '{raw_data_path}' for testing...")
    raw_file = np.load(raw_data_path, mmap_mode='r')
    raw_image = raw_file[args.sample_idx] 

    input_tensor = preprocess_input(raw_image).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        
        probabilities = F.softmax(outputs, dim=1)[0]
        
        pred_idx = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[pred_idx].item() * 100

    # 6. In ket qua
    print("\n================ AI PREDICTION RESULTS ================")
    print(f"AI predicts this drawing is: '{CLASSES[pred_idx].upper()}'")
    print(f"Model confidence: {confidence:.2f}%")
    print(f"Ground Truth Label: '{args.class_name.upper()}'")

    print("\n TOP 3:")
    top_prob, top_idx = torch.topk(probabilities, 3)
    for i in range(3):
        print(f"   {i+1}. {CLASSES[top_idx[i]].ljust(15)} : {top_prob[i].item()*100:.2f}%")
    print("========================================================")

if __name__ == "__main__":
    main()