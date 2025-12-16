import streamlit as st
import cv2
import numpy as np
import tempfile
import time

# =========================
# Streamlit設定
# =========================
st.set_page_config(page_title="転倒検知（天井対応）", layout="wide")
st.title("📹 転倒検知システム（平面・天井カメラ対応）")

st.markdown("""
### 判定ロジック
- 人の **移動量（速度）**
- **急激な移動 → 長時間停止**
で転倒を検知します。
""")

# =========================
# 動画アップロード
# =========================
uploaded_file = st.file_uploader(
    "天井カメラの動画をアップロードしてください",
    type=["mp4", "avi", "mov"]
)

# =========================
# 人検出（HOG）
# =========================
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

frame_area = st.image([])
status_area = st.empty()

# =========================
# パラメータ
# =========================
MOVE_THRESHOLD = 40        # 急激な移動量
STOP_TIME_THRESHOLD = 3.0 # 秒

prev_center = None
fall_candidate_time = None

# =========================
# 動画処理
# =========================
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("動画を開けません")
        st.stop()

    st.success("解析を開始します")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))

        boxes, _ = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )

        current_center = None

        for (x, y, w, h) in boxes:
            cx = x + w // 2
            cy = y + h // 2
            current_center = (cx, cy)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.circle(frame, current_center, 5, (0, 0, 255), -1)

        # ===== 動き解析 =====
        if prev_center and current_center:
            move_dist = np.linalg.norm(
                np.array(current_center) - np.array(prev_center)
            )

            # 急激な移動 → 転倒候補
            if move_dist > MOVE_THRESHOLD:
                fall_candidate_time = time.time()

        # ===== 停止時間判定 =====
        if fall_candidate_time:
            if current_center and prev_center:
                still_dist = np.linalg.norm(
                    np.array(current_center) - np.array(prev_center)
                )

                if still_dist < 5:
                    if time.time() - fall_candidate_time > STOP_TIME_THRESHOLD:
                        status_area.error("⚠️ 転倒を検知しました")
                else:
                    fall_candidate_time = None
        else:
            status_area.success("✅ 正常")

        prev_center = current_center

        frame_area.image(frame, channels="BGR")

    cap.release()
    st.info("解析終了")

else:
    st.info("動画をアップロードしてください")
