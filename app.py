import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# 1. Sayfa Yapılandırması
st.set_page_config(
    page_title="CookieVision | Bisküvi Kusur Tespiti",
    page_icon="🍪",
    layout="wide"
)

# 2. CSS Tasarımı
custom_css = """
<style>
    .stApp {
        background-color: #fcf9f2 !important;
        background-image: radial-gradient(#d4a373 0.75px, transparent 0.75px), radial-gradient(#faedcd 0.75px, #fcf9f2 0.75px) !important;
        background-size: 30px 30px !important;
        background-position: 0 0, 15px 15px !important;
        color: #1a1a1a !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #f5ede4 !important;
        border-right: 1px solid #e0d0c1 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #2c1810 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    h1, h2, h3 {
        color: #4a2c11 !important;
        font-weight: 700 !important;
    }

    div.stButton > button {
        background-color: #c97a3e !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }
    
    div.stButton > button:hover {
        background-color: #a85d26 !important;
        color: #ffffff !important;
        transform: scale(1.01);
    }

    .stAlert {
        border-radius: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. Model Yükleme (GitHub Release Direkt İndirme)
@st.cache_resource
def load_defect_model():
    model_path = "best_biscuit_patch_model.pth"
    
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        MODEL_URL = "https://github.com/boradogann/Cracked-Cookie-Analysis/releases/download/v1.0/best_biscuit_patch_model.pth"
        
        with st.spinner("Model ağırlıkları indiriliyor, lütfen bekleyin..."):
            urllib.request.urlretrieve(MODEL_URL, model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_defect_model()

# 4. Patch Dönüşümleri
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

PATCH_SIZE = 64

# 5. Kusur Analiz Fonksiyonu
def analyze_biscuit(image, nok_threshold=0.30, min_component_area=80):
    if image is None:
        return None, False, "Lütfen bir görsel yükleyin."

    img_np = np.array(image.convert("RGB"))
    h, w, _ = img_np.shape

    # Convex Hull Gövde Tespiti
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

    patch_batch = []
    coords = []

    STRIDE_VAL = 16 

    for y in range(0, h - PATCH_SIZE + 1, STRIDE_VAL):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE_VAL):
            roi_patch = solid_roi_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            if np.mean(roi_patch) >= 0.25:
                p = img_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_pil = Image.fromarray(p)
                patch_batch.append(transform(patch_pil))
                coords.append((y, x))

    if len(patch_batch) == 0:
        return img_np, False, "Görselde bisküvi tespit edilemedi!"

    batch_tensors = torch.stack(patch_batch).to(device)
    with torch.no_grad():
        outputs = model(batch_tensors)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

    for (y, x), prob in zip(coords, probs):
        heatmap_acc[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += prob
        count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1.0

    count_map[count_map == 0] = 1.0
    heatmap_avg = (heatmap_acc / count_map) * solid_roi_mask

    # İkili Eşikleme ve Morfolojik Kapanış
    binary_defects = (heatmap_avg >= nok_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closed_defects = cv2.morphologyEx(binary_defects, cv2.MORPH_CLOSE, kernel)

    # Bağlı Bileşen Analizi
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(closed_defects, connectivity=8)

    clean_final_mask = np.zeros((h, w), dtype=np.uint8)
    valid_defects_found = False

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            clean_final_mask[labels_im == i] = 255
            valid_defects_found = True

    status_str = "🔴 SONUÇ: KUSURLU (NOK)" if valid_defects_found else "🟢 SONUÇ: SAĞLAM (OK)"

    # Görselleştirme Katmanı
    overlay = img_np.copy()
    if valid_defects_found:
        soft_mask = cv2.GaussianBlur(clean_final_mask.astype(np.float32) / 255.0, (7, 7), 0)
        red_tint = np.zeros_like(img_np)
        red_tint[:, :] = [255, 30, 30]

        alpha = (soft_mask[:, :, None] * 0.65)
        overlay = (img_np * (1.0 - alpha) + red_tint * alpha).astype(np.uint8)

    return overlay, valid_defects_found, status_str

# 6. Sidebar Kontrolleri
st.sidebar.title("🍪 Kontrol Paneli")

input_mode = st.sidebar.radio(
    "Görsel Kaynağı:",
    ["Örnek Görsellerden Seç", "Kendi Fotoğrafını Yükle", "📸 Kameradan Fotoğraf Çek"]
)

selected_image = None

if input_mode == "Örnek Görsellerden Seç":
    samples_dir = "test"
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    sample_dict = {}

    if os.path.exists(samples_dir):
        for root, _, files in os.walk(samples_dir):
            if "masks" in root:
                continue
            for f in files:
                if f.lower().endswith(valid_exts):
                    rel_path = os.path.relpath(os.path.join(root, f), samples_dir)
                    display_name = f"[{os.path.dirname(rel_path)}] {f}" if os.path.dirname(rel_path) else f
                    sample_dict[display_name] = os.path.join(root, f)

    if sample_dict:
        options = sorted(list(sample_dict.keys()))
        chosen_display = st.sidebar.radio("Test Görseli Seçin:", options)
        selected_image = Image.open(sample_dict[chosen_display])
    else:
        st.sidebar.warning("`test/` klasöründe görsel bulunamadı.")
elif input_mode == "Kendi Fotoğrafını Yükle":
    uploaded_file = st.sidebar.file_uploader("Bisküvi fotoğrafı yükleyin...", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
else:
    camera_photo = st.sidebar.camera_input("Kameradan fotoğraf çekin")
    if camera_photo is not None:
        selected_image = Image.open(camera_photo)

st.sidebar.divider()
st.sidebar.subheader("Model Ayarları")
threshold_slider = st.sidebar.slider("NOK Hassasiyet Eşiği", min_value=0.2, max_value=0.8, value=0.70, step=0.05)

# 7. Ana Ekran
st.title("🍪 Bisküvi Kalite Kontrol Sistemi")
st.write("Yapay zeka tabanlı yüzey hasarı, kırık ve çatlak tespit arayüzü.")

if selected_image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Girdi Görseli")
        st.image(selected_image, use_container_width=True)

    with col2:
        st.subheader("Kusur Analiz Sonucu")
        if st.button("🔍 Görseli Analiz Et", type="primary", use_container_width=True):
            with st.spinner("Model analiz ediyor..."):
                result_img, is_defective, status_str = analyze_biscuit(
                    selected_image,
                    nok_threshold=threshold_slider,
                    min_component_area=80
                )
            st.image(result_img, use_container_width=True)
            st.divider()
            if is_defective:
                st.error(status_str)
            else:
                st.success(status_str)
        else:
            st.info("Analizi başlatmak için yukarıdaki **'Görseli Analiz Et'** butonuna basın.")
else:
    st.info("👈 Lütfen sol panelden bir test görseli seçin, fotoğraf yükleyin veya kameranızı kullanın.")
