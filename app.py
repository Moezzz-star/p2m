import os
import hashlib
import requests
from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ------------------------------
# Flask setup
# ------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ------------------------------
# Télécharger le modèle s'il n'existe pas
# ------------------------------
MODEL_DIR = "models"
MODEL_FILENAME = "cnn_attention_model4endo(ratio).pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://huggingface.co/zaazazzz/cnnattentionmodel4endoratio/resolve/main/cnn_attention_model4endo(ratio).pth"
EXPECTED_HASH = "c082c841702c0917268b4a58aa48e186e598f501b22bd81e65cab27ac2c24711"

def check_sha256(filepath, expected_sha256):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_sha256

def download_model():
    print("📦 Téléchargement du modèle depuis Hugging Face...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("✅ Modèle téléchargé avec succès.")

if not os.path.exists(MODEL_PATH):
    download_model()

if not check_sha256(MODEL_PATH, EXPECTED_HASH):
    raise ValueError("❌ Le fichier modèle est corrompu ou modifié. SHA-256 invalide.")

print("✅ Fichier modèle vérifié avec succès. SHA-256 OK.")

# ------------------------------
# Chargement du modèle PyTorch
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.to(device)
model.eval()

idx_to_label = {0: 'High', 1: 'Low', 2: 'Medium'}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------------
# Utils
# ------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(filepath):
    image = Image.open(filepath).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    prob_dict = {idx_to_label[i]: float(probs[i]) for i in range(3)}
    pred_label = max(prob_dict, key=prob_dict.get)
    return pred_label, prob_dict

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', error='No selected file or invalid format')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    pred_label, prob_dict = predict_image(filepath)

    return render_template(
        'result.html',
        filename=filename,
        pred_label=pred_label,
        prob_dict=prob_dict
    )

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

# ------------------------------
# Run
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
