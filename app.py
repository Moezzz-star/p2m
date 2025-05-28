import os
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
# Télécharger le modèle depuis Hugging Face si absent
# ------------------------------
MODEL_FILENAME = 'cnn_attention_model4endo(ratio).pth'
MODEL_PATH = os.path.join('models', MODEL_FILENAME)
MODEL_URL = 'https://huggingface.co/zaazazzz/cnnattentionmodel4endoratio/blob/main/cnn_attention_model4endo(ratio).pth'  # 🔁 Remplace ici

os.makedirs('models', exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    print(f"📦 Téléchargement du modèle depuis Hugging Face : {MODEL_URL}")
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("✅ Modèle téléchargé avec succès.")
    else:
        raise RuntimeError(f"❌ Échec du téléchargement ({response.status_code})")

# ------------------------------
# Chargement du modèle
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

idx_to_label = {0: 'High', 1: 'Low', 2: 'Medium'}

# ------------------------------
# Transformations d'image
# ------------------------------
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
    """Retourne (label_str, {label: prob})"""
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

