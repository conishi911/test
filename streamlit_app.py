import streamlit as st
import cv2
import numpy as np
import tempfile
import time

st.set_page_config(page_title="転倒検知＋倒れ時間計測", layout="wide")
st.title("📹 転倒検知システム（部分遮蔽対応・倒れ時間計測）")

uploaded_file = st.file_uploader(
    "動画をアップロードしてください",
    type=["mp4", "avi", "mov"]
)

frame_area = st.image([])
status_area = st.empty()
time_area = st.empty()

# =========================
# パラメータ
# =========================
MOVEMENT_THRESHOLD = 3       # 光学フローの閾値
STOP_TIME_THRESHOLD = 1.0    # 秒（倒れ開始とみなす最低時間）

# =========================
# 状態変数
# =========================
prev_gray = None
still_start_time = None
fallen = False
fall_start_time = None

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        movement = 0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            movement = np.mean(mag)

            # ===== 転倒判定 =====
            if movement < MOVEMENT_THRESHOLD:
                if still_start_time is None:
                    still_start_time = time.time()
                elif time.time() - still_start_time > STOP_TIME_THRESHOLD:
                    if not fallen:
                        # 倒れ状態開始
                        fall_start_time = time.time()
                        fallen = True
            else:
                # 動いている → 倒れ解除
                still_start_time = None
                fallen = False
                fall_start_time = None

        # ===== 表示 =====
        if fallen:
            status_area.error("⚠️ 転倒を検知しました（部分遮蔽対応）")
            elapsed = time.time() - fall_start_time
            time_area.info(f"倒れている時間: {elapsed:.1f} 秒")
        else:
            status_area.success("✅ 正常")
            time_area.empty()

        prev_gray = gray
        frame_area.image(frame, channels="BGR")

    cap.release()
    st.info("解析終了")

else:
    st.info("動画をアップロードしてください")
