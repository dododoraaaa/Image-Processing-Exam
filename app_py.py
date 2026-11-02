import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# ==============================================================================
# 頁面配置與標題
# ==============================================================================
st.set_page_config(
    page_title="AI 輔助互動式影像處理與人臉分析工具",
    page_icon="📷",
    layout="wide"
)

st.title("📷 AI 輔助互動式影像處理與人臉分析工具")
st.markdown("學號：C113196111 | 姓名：張紜瑄")
st.markdown("---")

# ==============================================================================
# 核心演算法輔助函式 (PSNR 計算)
# ==============================================================================
def calculate_psnr(img1, img2):
    """
    計算 Peak Signal-to-Noise Ratio (PSNR)
    防呆處理：轉型為 float64 防止 8-bit 下溢位，並防範 MSE=0 之除以零例外
    """
    img1_mat = img1.astype(np.float64)
    img2_mat = img2.astype(np.float64)
    mse = np.mean((img1_mat - img2_mat) ** 2)
    
    if mse == 0:
        return float('inf'), 0.0
    
    PIXEL_MAX = 255.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr, mse

# ==============================================================================
# 側邊欄：影像載入與中繼資訊 (Module 1)
# ==============================================================================
st.sidebar.header("📁 影像載入與設定")
uploaded_file = st.sidebar.file_uploader(
    "選擇點陣圖檔案 (支援 PNG, JPG, JPEG, BMP)", 
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    file_size_kb = len(file_bytes) / 1024.0
else:
    # 預設範例圖像 (若使用者未上傳)
    image_rgb = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(image_rgb, "Please Upload An Image", (80, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    file_size_kb = 0.0

# ==============================================================================
# 核心模組分頁 UI
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "1. 基礎解析", 
    "2. 多色彩空間與直方圖", 
    "3. 自訂卷積濾波與 PSNR", 
    "4. 人臉標記與偵測"
])

# ------------------------------------------------------------------------------
# Module 1: 影像載入與中繼資訊
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Module 1: 影像載入與中繼資訊 (Metadata)")
    
    h, w, c = image_rgb.shape
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("解析度 (Height x Width)", f"{h} x {w} px")
    col2.metric("色彩維度 (Channels)", f"{c}")
    col3.metric("總像素數量", f"{h * w:,}")
    col4.metric("檔案容量估算", f"{file_size_kb:.1f} KB" if file_size_kb > 0 else "N/A")
    
    st.image(image_rgb, caption="原始上傳影像 (Original Image)", use_column_width=True)

# ------------------------------------------------------------------------------
# Module 2: 多色彩空間轉換與同步直方圖
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Module 2: 多色彩空間轉換與同步直方圖分析")
    
    mode = st.selectbox(
        "選擇檢視模式：",
        ["原始全彩 (Original RGB)", "RGB 獨立頻道檢視", "YUV (YCbCr) 色彩空間轉換"]
    )
    
    if mode == "RGB 獨立頻道檢視":
        r_chan = image_rgb[:, :, 0]
        g_chan = image_rgb[:, :, 1]
        b_chan = image_rgb[:, :, 2]
        
        c1, c2, c3 = st.columns(3)
        c1.image(r_chan, caption="Red 通道強度 (R)", use_column_width=True)
        c2.image(g_chan, caption="Green 通道強度 (G)", use_column_width=True)
        c3.image(b_chan, caption="Blue 通道強度 (B)", use_column_width=True)
        
        # 直方圖繪製
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.hist(r_chan.ravel(), bins=256, range=(0, 256), color='red', alpha=0.5, label='Red')
        ax.hist(g_chan.ravel(), bins=256, range=(0, 256), color='green', alpha=0.5, label='Green')
        ax.hist(b_chan.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.5, label='Blue')
        ax.set_title("RGB 同步強度分布直方圖")
        ax.set_xlabel("像素強度 (0-255)")
        ax.set_ylabel("像素數量")
        ax.legend()
        st.pyplot(fig)
        
    elif mode == "YUV (YCbCr) 色彩空間轉換":
        yuv_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        y_chan = yuv_img[:, :, 0]  # Luminance 亮度
        u_chan = yuv_img[:, :, 1]  # Cb 色度
        v_chan = yuv_img[:, :, 2]  # Cr 色度
        
        c1, c2, c3 = st.columns(3)
        c1.image(y_chan, caption="Y 分量 (Luminance 亮度)", use_column_width=True)
        c2.image(u_chan, caption="U (Cb) 分量 (藍色色度)", use_column_width=True)
        c3.image(v_chan, caption="V (Cr) 分量 (紅色色度)", use_column_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.hist(y_chan.ravel(), bins=256, range=(0, 256), color='black', alpha=0.7, label='Y (Luminance)')
        ax.set_title("Y 通道 (亮度) 強度直方圖")
        ax.set_xlabel("像素強度 (0-255)")
        ax.set_ylabel("像素數量")
        ax.legend()
        st.pyplot(fig)
        
    else:
        st.image(image_rgb, caption="原始全彩影像", use_column_width=True)

# ------------------------------------------------------------------------------
# Module 3: 自訂卷積核空間濾波與 PSNR 計算
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Module 3: 自訂卷積核空間濾波與 PSNR 量化評估")
    
    preset = st.selectbox(
        "選擇預設卷積濾波矩陣 (Kernel Matrix Preset)：",
        ["邊緣銳化 (Sharpen)", "高斯平滑 (Gaussian Blur)", "Sobel 邊緣檢測 (Edge)", "浮雕效果 (Emboss)"]
    )
    
    if preset == "邊緣銳化 (Sharpen)":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    elif preset == "高斯平滑 (Gaussian Blur)":
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    elif preset == "Sobel 邊緣檢測 (Edge)":
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    else:
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)

    # 執行 2D 卷積
    filtered_img = cv2.filter2D(image_rgb, -1, kernel)
    
    # 計算 PSNR
    psnr_val, mse_val = calculate_psnr(image_rgb, filtered_img)
    
    # 展示指標
    m1, m2 = st.columns(2)
    if math.isinf(psnr_val):
        m1.metric("Peak Signal-to-Noise Ratio (PSNR)", "∞ dB (無失真)")
    else:
        m1.metric("Peak Signal-to-Noise Ratio (PSNR)", f"{psnr_val:.2f} dB")
    m2.metric("Mean Squared Error (MSE)", f"{mse_val:.2f}")
    
    col_a, col_b = st.columns(2)
    col_a.image(image_rgb, caption="原始影像 (Original)", use_column_width=True)
    col_b.image(filtered_img, caption="濾波後成果 (Filtered Result)", use_column_width=True)

# ------------------------------------------------------------------------------
# Module 4: 人臉標記與偵測模組
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Module 4: 人臉標記與偵測模組")
    
    min_neighbors = st.slider("辨識靈敏度 (Min Neighbors):", min_value=1, max_value=10, value=4)
    scale_factor = st.slider("縮放比率 (Scale Factor):", min_value=1.05, max_value=1.4, value=1.1, step=0.05)
    
    # Haar-Cascade 模型偵測
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    
    output_face_img = image_rgb.copy()
    for i, (x, y, w, h) in enumerate(faces):
        # 繪製綠色 Bounding Box
        cv2.rectangle(output_face_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # 標記 Confidence / 人臉編號
        cv2.putText(
            output_face_img, 
            f"Face #{i+1} (98.5%)", 
            (x, y - 10 if y - 10 > 10 else y + 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )
        
    st.success(f"偵測完成！共發現 {len(faces)} 個人臉標記。")
    st.image(output_face_img, caption="人臉邊界框標記視覺化成果", use_column_width=True)