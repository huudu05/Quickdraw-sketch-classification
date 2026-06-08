import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.config import CLASSES
from src.model import QuickDrawCNN
from src.utils import preprocess_ui_image

WINDOW_NAME = "Quick Draw Application"
CANVAS_SIZE = 448
drawing = False
ix, iy = -1, -1

CONFIDENCE_THRESHOLD = 70.0

canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

current_prediction = "Let's draw..."

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, canvas

    model = param['model']
    device = param['device']
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (ix, iy), (x, y), 255, thickness=8)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(canvas, (ix, iy), (x, y), 255, thickness=8)
        predict_drawing(model, device)

def predict_drawing(model, device):
    global canvas, current_prediction

    if np.sum(canvas) == 0:
        return
    
    input_tensor = preprocess_ui_image(canvas).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(outputs, dim=1).item()

        raw_confidence = probabilities[pred_idx].item() * 100

        if raw_confidence >= CONFIDENCE_THRESHOLD:
            current_prediction = CLASSES[pred_idx].upper()
        else:
            current_prediction = "UNCERTAIN (KEEP DRAWING...)"

    

def main():
    global canvas, current_prediction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "checkpoints/best_model.pth"

    if not os.path.exists(model_path):
        return

    model = QuickDrawCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    print("[*] Sketch Recognition Application is ready!")
    print("    - Hold and drag the LEFT MOUSE BUTTON to draw.")
    print("    - Press 'C' to clear the canvas.")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(
        WINDOW_NAME, 
        draw_circle,
        {
            'model': model,
            'device': device
        }
    )

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        display_screen = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(display_screen, (0, CANVAS_SIZE - 40), (CANVAS_SIZE, CANVAS_SIZE), (50, 50, 50), -1)

        status_text = f"AI: {current_prediction}"
        cv2.putText(display_screen, status_text, (10, CANVAS_SIZE - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow(WINDOW_NAME, display_screen)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord('C'):
            canvas[:, :] = 0
            current_prediction = "Let's draw..."
       
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
