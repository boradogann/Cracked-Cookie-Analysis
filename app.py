import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Sayfa Yapılandırması
st.set_page_config(
    page_title="CookieVision | Bisküvi Kusur Tespiti",
    page_icon="🍪",
    layout="wide"
)

# 1. Zarif Kurabiye Temalı CSS
custom_css = """
<style>
    /* Arka plana şık, hafif ve yarı saydam kurabiye deseni */
    .stApp {
        background-color: #fcf9f2;
        background-image: radial-gradient(#d4a373 0.75px, transparent 0.75px), radial-gradient(#faedcd 0.75px, #fcf9f2 0.75px);
        background-size: 30px 30px;
        background-position: 0 0, 15px 15px;
    }
    
    /* Sidebar Stili */
    section[data-testid="stSidebar"] {
        background-color: #f4ede2;
        border-right: 1px solid #e6ccb2;
    }

    /* Başlık ve Kart Tasarımları */
    h1, h2, h3 {
        color: #6b4423 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stAlert {
        border-radius: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 2. Model Yükleme
def load_defect_model():
    model_path = "best_biscuit_patch_model.pth"
    
    # Model yerelde yoksa Google Drive'dan otomatik indir
    if not os.path.exists(model_path):
        file_id = "1Iy6AAn5qN5sdxbumoELCpd09Z-EFwRFo"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device
# 3. Patch Dönüşümleri
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

PATCH_SIZE = 64
STRIDE_VAL = 16

# 4. Analiz Fonksiyonu
def analyze_biscuit(image, nok_threshold=0.30, min_component_area=80):
    img_np = np.array(image.convert("RGB"))
    h, w, _ = img_np.shape

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_roi_mask = np.zeros((h, w), dtype=np.uint8)

    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_cnt)
        cv2.drawContours(solid_roi_mask, [hull], -1, 1, thickness=-1)
    else:
        solid_roi_mask = (binary > 0).astype(np.uint8)

    heatmap_acc = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    patch_batch, coords = [], []

    for y in range(0, h - PATCH_SIZE + 1, STRIDE_VAL):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE_VAL):
            roi_patch = solid_roi_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if np.mean(roi_patch) >= 0.25:
                p = img_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_batch.append(transform(Image.fromarray(p)))
                coords.append((y, x))

    if len(patch_batch) == 0:
        return img_np, False, "Bisküvi tespit edilemedi!"

    batch_tensors = torch.stack(patch_batch).to(device)
    with torch.no_grad():
        outputs = model(batch_tensors)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

    for (y, x), prob in zip(coords, probs):
        heatmap_acc[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += prob
        count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1.0

    count_map[count_map == 0] = 1.0
    heatmap_avg = (heatmap_acc / count_map) * solid_roi_mask

    binary_defects = (heatmap_avg >= nok_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closed_defects = cv2.morphologyEx(binary_defects, cv2.MORPH_CLOSE, kernel)

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(closed_defects, connectivity=8)
    clean_final_mask = np.zeros((h, w), dtype=np.uint8)
    valid_defects_found = False

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            clean_final_mask[labels_im == i] = 255
            valid_defects_found = True

    overlay = img_np.copy()
    if valid_defects_found:
        soft_mask = cv2.GaussianBlur(clean_final_mask.astype(np.float32) / 255.0, (7, 7), 0)
        red_tint = np.zeros_like(img_np)
        red_tint[:, :] = [255, 30, 30]

        alpha = soft_mask[:, :, None] * 0.65
        overlay = (img_np * (1.0 - alpha) + red_tint * alpha).astype(np.uint8)

    return overlay, valid_defects_found, "Analiz Tamamlandı"

# 5. Sidebar Kontrolleri
st.sidebar.title("🍪 Kontrol Paneli")

# Görsel Seçim Yöntemi
input_mode = st.sidebar.radio(
    "Görsel Kaynağı:",
    ["Örnek Görsellerden Seç", "Kendi Fotoğrafını Yükle"]
)

selected_image = None

if input_mode == "Örnek Görsellerden Seç":
    samples_dir = "test"
    os.makedirs(samples_dir, exist_ok=True)
    sample_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if sample_files:
        chosen_sample = st.sidebar.selectbox("Test İçin Hazır Görsel Seçin:", sorted(sample_files))
        selected_image = Image.open(os.path.join(samples_dir, chosen_sample))
    else:
        st.sidebar.warning("Henüz `samples/` klasörüne örnek fotoğraf eklenmedi.")
else:
    uploaded_file = st.sidebar.file_uploader("Bisküvi fotoğrafı yükleyin...", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)

st.sidebar.divider()
st.sidebar.subheader("Model Hassasiyet Ayarları")
threshold_slider = st.sidebar.slider("NOK Hassasiyet Eşiği", min_value=0.1, max_value=0.8, value=0.30, step=0.05)
min_area_slider = st.sidebar.slider("Min. Kusur Alanı (Piksel)", min_value=30, max_value=300, value=80, step=10)

# 6. Ana Ekran
st.title("🍪 Bisküvi Kalite Kontrol Sistemi")
st.write("Yapay zeka tabanlı yüzey hasarı, kırık ve baskı kusuru tespit arayüzü.")

if selected_image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Girdi Görseli")
        st.image(selected_image, use_container_width=True)

    with st.spinner("Model analiz ediyor..."):
        result_img, is_defective, msg = analyze_biscuit(
            selected_image,
            nok_threshold=threshold_slider,
            min_component_area=min_area_slider
        )

    with col2:
        st.subheader("Kusur Analiz Sonucu")
        st.image(result_img, use_container_width=True)

    st.divider()
    if is_defective:
        st.error("🔴 SONUÇ: KUSURLU (NOK) - Hasarlı bölge kırmızıyla işaretlendi.")
    else:
        st.success("🟢 SONUÇ: SAĞLAM (OK) - Ürün sağlam standartlara uygun.")
else:
    st.info("👈 Lütfen sol panelden hazır bir örnek seçin veya test etmek için yeni bir fotoğraf yükleyin.")
