import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import gdown

st.set_page_config(
    page_title="CookieVision | Bisküvi Kusur Tespiti",
    page_icon="🍪",
    layout="wide"
)

# 1. Açık Renkli, Modern ve Okunaklı CSS
custom_css = """
<style>
    /* Ana Arka Plan */
    .stApp {
        background-color: #fcf9f2;
        background-image: radial-gradient(#d4a373 0.75px, transparent 0.75px), radial-gradient(#faedcd 0.75px, #fcf9f2 0.75px);
        background-size: 30px 30px;
        background-position: 0 0, 15px 15px;
        color: #1a1a1a !important;
    }
    
    /* Sidebar Arka Planı */
    section[data-testid="stSidebar"] {
        background-color: #f5ede4 !important;
        border-right: 1px solid #e0d0c1;
    }

    /* Genel Yazı Renkleri */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #2c1810 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    h1, h2, h3 {
        color: #4a2c11 !important;
        font-weight: 700 !important;
    }

    /* Selectbox (Seçim Kutusu) Açık Renk Düzeltmesi */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1.5px solid #d4a373 !important;
        border-radius: 8px !important;
    }

    /* Selectbox İçindeki Yazı ve Ok Simgesi */
    div[data-baseweb="select"] * {
        color: #1a1a1a !important;
        background-color: transparent !important;
    }

    /* Açılır Menü Listesi (Dropdown Menü) */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #ffffff !important;
    }
    
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: #faedcd !important;
        color: #4a2c11 !important;
    }

    /* Buton Tasarımı */
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
        transform: scale(1.02);
    }

    /* Slider / Bilgi Kutuları */
    .stAlert {
        border-radius: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 2. Model Yükleme
@st.cache_resource
def load_defect_model():
    model_path = "best_biscuit_patch_model.pth"
    
    if not os.path.exists(model_path):
        file_id = "13eY6048dG51DskZc49GZc6Y9b5E18f1p"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(id=file_id, output=model_path, quiet=False, fuzzy=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_defect_model()

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

# 5. Sidebar Kontrolleri ve Ayarlar
st.sidebar.title("🍪 Kontrol Paneli")

input_mode = st.sidebar.radio(
    "Görsel Kaynağı:",
    ["Örnek Görsellerden Seç", "Kendi Fotoğrafını Yükle"]
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
        options = ["--- Bir görsel seçin ---"] + sorted(list(sample_dict.keys()))
        chosen_display = st.sidebar.selectbox("Test Görseli Seçin:", options)
        if chosen_display != "--- Bir görsel seçin ---":
            selected_image = Image.open(sample_dict[chosen_display])
    else:
        st.sidebar.warning("`test/` klasöründe uygun görsel bulunamadı.")
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
st.write("Yapay zeka tabanlı yüzey hasarı, kırık ve çatlak tespit arayüzü.")

if selected_image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Seçilen Görsel")
        st.image(selected_image, use_container_width=True)

    with col2:
        st.subheader("Kusur Analiz Sonucu")
        if st.button("🔍 Görseli Analiz Et", type="primary", use_container_width=True):
            with st.spinner("Model analiz ediyor..."):
                result_img, is_defective, msg = analyze_biscuit(
                    selected_image,
                    nok_threshold=threshold_slider,
                    min_component_area=min_area_slider
                )
            st.image(result_img, use_container_width=True)
            st.divider()
            if is_defective:
                st.error("🔴 SONUÇ: KUSURLU (NOK) - Hasarlı bölge kırmızıyla işaretlendi.")
            else:
                st.success("🟢 SONUÇ: SAĞLAM (OK) - Ürün standartlara uygun.")
        else:
            st.info("Analizi başlatmak için yukarıdaki **'Görseli Analiz Et'** butonuna basın.")
else:
    st.info("👈 Lütfen sol panelden bir test görseli seçin veya kendi fotoğrafınızı yükleyin.")
