import onnxruntime as ort
import cv2
import numpy as np
import mss
import pyautogui
import pygetwindow as gw
from pynput import mouse

# Load ONNX model
model_path = "fortnite_model.onnx"  # Change to your model's file
session = ort.InferenceSession(model_path)

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Function to get Fortnite window's region
def get_fortnite_window_region():
    try:
        window = gw.getWindowsWithTitle("Fortnite")[0]  # Find the first window with "Fortnite" in the title
        return {
            "top": window.top,
            "left": window.left,
            "width": window.width,
            "height": window.height
        }
    except IndexError:
        print("Fortnite window not found!")
        return None

# Initialize screen capture (mss)
sct = mss.mss()

# Track right-click state
right_click_held = False

def on_press(button):
    global right_click_held
    if button == mouse.Button.right:
        right_click_held = True

def on_release(button):
    global right_click_held
    if button == mouse.Button.right:
        right_click_held = False

# Start mouse listener
mouse_listener = mouse.Listener(on_press=on_press, on_release=on_release)
mouse_listener.start()

def preprocess(frame):
    """Resize and normalize image for ONNX model"""
    frame = cv2.resize(frame, (640, 640))  # Adjust size to match the model's input
    frame = frame.astype(np.float32) / 255.0  # Normalize
    frame = np.transpose(frame, (2, 0, 1))  # Convert HWC to CHW
    return np.expand_dims(frame, axis=0)  # Add batch dimension

while True:
    # Get the Fortnite window region dynamically
    monitor = get_fortnite_window_region()
    if monitor is None:
        break  # Exit if Fortnite window is not found

    # Capture the screen for the Fortnite window region
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Preprocess
    input_data = preprocess(frame)

    # Run inference
    outputs = session.run([output_name], {input_name: input_data})[0]

    best_confidence = 0
    best_target = None

    # Process detections
    for detection in outputs[0]:
        x, y, w, h, confidence, class_id = detection[:6]
        if confidence > 0.5:  # Adjust confidence threshold if needed
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Track highest confidence detection
            if confidence > best_confidence:
                best_confidence = confidence
                best_target = ((x1 + x2) // 2, (y1 + y2) // 2)  # Center of the box

    # Move mouse when right-click is held
    if right_click_held and best_target:
        screen_x = monitor["left"] + best_target[0]
        screen_y = monitor["top"] + best_target[1]
        pyautogui.moveTo(screen_x, screen_y, duration=0.05)  # Smooth mouse movement

    # Show the frame
    cv2.imshow("Fortnite Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
