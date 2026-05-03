import os, io, base64, numpy as np, cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf

CLASSES  = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
IMG_SIZE = (224, 224)
SEUIL    = 0.70

print("Chargement du modele...")
MODEL_PATH = os.getenv("MODEL_PATH", "model/Modele_Riz.h5")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modele charge ✓")

app = FastAPI(title="API Classification de Riz", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), np.array(img)

def compute_gradcam(arr, img_orig, class_idx):
    try:
        x_var = tf.Variable(tf.cast(arr, tf.float32))
        with tf.GradientTape() as tape:
            tape.watch(x_var)
            preds = model(x_var, training=False)
            score = preds[:, class_idx]
        grads        = tape.gradient(score, x_var)
        heatmap      = tf.reduce_mean(tf.abs(grads[0]), axis=-1).numpy()
        heatmap      = heatmap / (heatmap.max() + 1e-8)
        heat_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heat_colored = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
        superposed   = cv2.addWeighted(img_orig, 0.6, heat_colored, 0.4, 0)
        _, buf       = cv2.imencode(".jpg", cv2.cvtColor(superposed, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode("utf-8")
    except:
        return ""

@app.get("/")
def root():
    return {"message": "API Classification de Riz", "classes": CLASSES}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image uniquement")
    image_bytes = await file.read()
    try:
        arr, img_orig = preprocess(image_bytes)
        preds         = model.predict(arr, verbose=0)[0]
        class_idx     = int(np.argmax(preds))
        confidence    = float(preds[class_idx]) * 100
        classe        = "Inconnu - pas du riz" if float(preds[class_idx]) < SEUIL else CLASSES[class_idx]
        gradcam       = compute_gradcam(arr, img_orig, class_idx)
        return JSONResponse({
            "classe"      : classe,
            "confiance"   : round(confidence, 2),
            "probabilites": {CLASSES[i]: round(float(p)*100, 2) for i, p in enumerate(preds)},
            "gradcam"     : gradcam
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
