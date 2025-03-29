import os
import io
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__, static_folder='static', static_url_path='')  


class MyModel(nn.Module):
    def __init__(self, num_chars, input_channels=1):
        super(MyModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )
        
        self.lstm_hidden_size = 256
        self.feature_projection = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        
        self.rnn_input_size = 256
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        self.classifier = nn.Linear(self.lstm_hidden_size * 2, num_chars)
    
    def forward(self, x):
        features = self.cnn(x)  # [batch, channels, height, width]
        features = self.feature_projection(features)
        features = features.mean(dim=2)  # average across height -> [batch, channels, width]
        features = features.permute(0, 2, 1)  # [batch, width, channels]
        rnn_output, _ = self.lstm(features)
        logits = self.classifier(rnn_output)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs
    
    def predict(self, x):
        log_probs = self.forward(x)
        predictions = torch.argmax(log_probs, dim=2)
        return predictions

# Example: a 77-character model
char_to_idx = {
    '!': 1, '"': 2, '#': 3, "'": 4, '(': 5, ')': 6, '*': 7, ',': 8, '-': 9, '.': 10, '/': 11,
    '0': 12, '1': 13, '2': 14, '3': 15, '4': 16, '5': 17, '6': 18, '7': 19, '8': 20, '9': 21,
    ':': 22, ';': 23, '?': 24, 'A': 25, 'B': 26, 'C': 27, 'D': 28, 'E': 29, 'F': 30, 'G': 31,
    'H': 32, 'I': 33, 'J': 34, 'K': 35, 'L': 36, 'M': 37, 'N': 38, 'O': 39, 'P': 40, 'Q': 41,
    'R': 42, 'S': 43, 'T': 44, 'U': 45, 'V': 46, 'W': 47, 'X': 48, 'Y': 49, 'Z': 50, 'a': 51,
    'b': 52, 'c': 53, 'd': 54, 'e': 55, 'f': 56, 'g': 57, 'h': 58, 'i': 59, 'j': 60, 'k': 61,
    'l': 62, 'm': 63, 'n': 64, 'o': 65, 'p': 66, 'q': 67, 'r': 68, 's': 69, 't': 70, 'u': 71,
    'v': 72, 'w': 73, 'x': 74, 'y': 75, 'z': 76, '<BLANK>': 0
}
idx_to_char = {v: k for k, v in char_to_idx.items()}

def decode_predictions(predictions, idx_to_char, blank_idx=0):
    """
    Given a sequence of predicted indices (one per time step), collapse duplicates
    and remove blank tokens to produce a final string.
    """
    results = []
    for pred in predictions:
        collapsed = []
        previous = -1
        for p in pred:
            token = p.item() if hasattr(p, 'item') else p
            if token != previous:
                collapsed.append(token)
            previous = token
        # Remove blank tokens (assuming blank index is 0) and map to characters.
        decoded = [idx_to_char.get(idx, '') for idx in collapsed if idx != blank_idx]
        results.append(''.join(decoded))
    return results

# IMAGE PREPROCESSING FUNCTION
def preprocess_image(img: np.ndarray, img_height: int = 32) -> torch.Tensor:
    # Convert to grayscale if needed.
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert the image if the average brightness is low.
    if np.mean(img) < 127:
        img = 255 - img
    h, w = img.shape
    new_h = img_height
    new_w = int(w * (new_h / h))
    img = cv2.resize(img, (new_w, new_h))
    img = img.astype(np.float32) / 255.0
    # Convert to tensor with shape [1, H, W]
    img_tensor = torch.FloatTensor(img).unsqueeze(0)
    return img_tensor


# LOAD THE PRETRAINED MODEL
state_dict_path = "/Users/zachliu/Documents/aps360_handwriting/model_weights/base_model_weight.pt"
model = MyModel(num_chars=77)
model.load_state_dict(torch.load(state_dict_path, map_location=torch.device("cpu")))
model.eval()


# FLASK ENDPOINTS
# Serve the front-end HTML 
@app.route('/', methods=['GET'])
def index():
    return send_from_directory(app.static_folder, 'index.html')

# /predict endpoint to receive an image and return the recognized word.
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # Accept the first file provided regardless of its key.
    if not request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = next(iter(request.files.values()))
    file_bytes = file.read()
    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Preprocess the image and add a batch dimension.
    input_tensor = preprocess_image(img).unsqueeze(0)
    with torch.no_grad():
        # Use the model's predict method to get indices.
        predictions = model.predict(input_tensor)
    # Decode the predictions to obtain a word.
    decoded_texts = decode_predictions(predictions, idx_to_char, blank_idx=0)
    recognized_text = decoded_texts[0] if decoded_texts else ""
    
    # Return only the recognized text as the final result.
    return jsonify({
        'recognized_text': recognized_text
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
