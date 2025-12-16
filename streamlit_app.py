import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from collections import deque

# =========================
# Streamlit設定
# =========================
st.set_page_config(page_title="転倒検知（歩行対応）", layout="wide")
st.title("📹 転倒検知システム（平面・天井・歩行対応）")

st.markdown("""
### 判定ロジック（実運用レベル）
1. **移動量（速度）を常時計測**
2. **急減速を転倒トリガー** とする
3. **低移動状態が継続** → 転倒確定
""")

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
# パラメータ（重要）
# =========================
LOW_MOVE_THRESHOLD = 8        # ほぼ止まっている
HIGH_MOVE_THRESHOLD = 25     # 歩行
DECEL_THRESHOLD = 15         # 急減速
CONFIRM_TIME = 2.5           # 秒

speed_history = deque(maxlen=5)

prev_center = None
fall_trigger_time = None
fallen = False

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

        if len(boxes) > 0:
            x, y, w, h = boxes[0]
            current_center = (x + w // 2, y + h // 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.circle(frame, current_center, 5, (0, 0, 255), -1)

        # ===== 速度計算 =====
        speed = 0
        if prev_center and current_center:
            speed = np.linalg.norm(
                np.array(current_center) - np.array(prev_center)
            )

        speed_history.append(speed)
        avg_speed = np.mean(speed_history) if speed_history else 0

        # ===== 転倒トリガー（急減速）=====
        if avg_speed > HIGH_MOVE_THRESHOLD:
            walking = True
        else:
            walking = False

        if walking and avg_speed < DECEL_THRESHOLD:
            fall_trigger_time = time.time()

        # ===== 転倒確定判定 =====
        if fall_trigger_time:
            if avg_speed < LOW_MOVE_THRESHOLD:
                if time.time() - fall_trigger_time > CONFIRM_TIME:
                    fallen = True
            else:
                # 再び歩いたらリセット
                fall_trigger_time = None
                fallen = False

        # ===== 表示 =====
        if fallen:
            status_area.error("⚠️ 歩行中の転倒を検知しました")
        else:
            status_area.success("✅ 正常")

        prev_center = current_center
        frame_area.image(frame, channels="BGR")

    cap.release()
    st.info("解析終了")

else:
    st.info("動画をアップロードしてください")
