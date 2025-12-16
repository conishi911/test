import streamlit as st
import cv2
import numpy as np
import tempfile

# =========================
# Streamlit設定
# =========================
st.set_page_config(
    page_title="転倒検知システム",
    layout="wide"
)

st.title("📹 人体転倒検知システム（Streamlit Cloud対応）")
st.markdown(
    """
    **人物の縦横比**を用いて  
    **倒れている / 倒れていない** を判定します。
    """
)

# =========================
# 動画アップロード
# =========================
uploaded_file = st.file_uploader(
    "解析する動画ファイルをアップロードしてください",
    type=["mp4", "avi", "mov"]
)

# =========================
# 人物検出器（HOG）
# =========================
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

status_area = st.empty()
frame_area = st.image([])

# =========================
# 動画処理
# =========================
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("動画を開けませんでした")
        st.stop()

    st.success("動画解析を開始します")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # リサイズ（高速化）
        frame = cv2.resize(frame, (640, 360))

        boxes, _ = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        fallen = False

        for (x, y, w, h) in boxes:
            aspect_ratio = w / h

            if aspect_ratio > 1.2:
                fallen = True
                color = (0, 0, 255)
                label = "FALL DETECTED"
            else:
                color = (0, 255, 0)
                label = "NORMAL"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # ステータス表示
        if fallen:
            status_area.error("⚠️ 人が倒れています")
        else:
            status_area.success("✅ 正常姿勢")

        frame_area.image(frame, channels="BGR")

    cap.release()
    st.info("解析が終了しました")

else:
    st.info("左のエリアから動画をアップロードしてください")
