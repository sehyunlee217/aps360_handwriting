import os
import subprocess
from PIL import Image, ImageOps
from rapidfuzz.distance import Levenshtein 


input_dir = "/Users/zachliu/Downloads/baseline_testdata"
output_dir = "/Users/zachliu/Downloads/baseline_testdata/preprocessed"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        input_path = os.path.join(input_dir, filename)

        # Preprocessing
        with Image.open(input_path) as img:
            img = img.convert("L")  # Convert to grayscale
            img = img.resize((128,32), Image.Resampling.LANCZOS)  # Resize to 128x32
            preprocessed_path = os.path.join(output_dir, filename)
            img.save(preprocessed_path)


# Step 1: Your ground truth labels (image filename → actual text)
ground_truth = {
    "img1.png": "to",
    "img2.png": "negotiate",
    "img3.png": "terms",
    "img4.png": "for",
    "img5.png": "and",
    "img6.png": ".",
    "img7.png": "Market",
    "img8.png": "Common",
    "img9.png": "the",
    "img10.png": "joining",
}

# Step 2: Read OCR output from Tesseract
output_dir = "/Users/zachliu/Downloads/baseline_testdata/preprocessed"
results = {}
for filename in ground_truth:
    pred_file = os.path.join(output_dir, filename.replace(".png", ".txt"))
    if os.path.exists(pred_file):
        with open(pred_file, "r") as f:
            prediction = f.read().strip()
        results[filename] = prediction
    else:
        results[filename] = ""

# Step 3: Calculate CER and WER
total_chars, total_char_errors = 0, 0
total_words, total_word_errors = 0, 0

for file, true_text in ground_truth.items():
    pred_text = results.get(file, "")

    # CER (character-level Levenshtein distance)
    char_errors = Levenshtein.distance(true_text, pred_text)
    total_char_errors += char_errors
    total_chars += len(true_text)

    # WER (word-level Levenshtein distance)
    true_words = true_text.split()
    pred_words = pred_text.split()
    word_errors = Levenshtein.distance(" ".join(true_words), " ".join(pred_words))
    total_word_errors += word_errors
    total_words += len(true_words)

# Step 4: Print accuracy
cer = total_char_errors / total_chars if total_chars > 0 else 0
wer = total_word_errors / total_words if total_words > 0 else 0

print(f"Character Error Rate (CER): {cer:.2%}")
print(f"Word Error Rate (WER): {wer:.2%}")
