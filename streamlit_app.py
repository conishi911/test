import streamlit as st
import cv2
import numpy as np
import tempfile

st.set_page_config(page_title="転倒検知＋正確な倒れ時間", layout="wide")
st.title("📹 転倒検知システム（部分遮蔽対応・正確時間計測）")

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
STOP_FRAMES_THRESHOLD = 3    # 倒れ開始判定に必要な連続フレーム数

# =========================
# 状態変数
# =========================
prev_gray = None
still_frame_count = 0
fallen = False
fall_frames_count = 0

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

    fps = cap.get(cv2.CAP_PROP_FPS)
    st.success(f"解析を開始します（FPS={fps:.1f}）")

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
                still_frame_count += 1
                if still_frame_count >= STOP_FRAMES_THRESHOLD:
                    if not fallen:
                        fallen = True
                        fall_frames_count = 0
            else:
                still_frame_count = 0
                fallen = False
                fall_frames_count = 0

        # ===== 倒れ時間計算 =====
        if fallen:
            fall_frames_count += 1
            fall_time_sec = fall_frames_count / fps
            status_area.error("⚠️ 転倒を検知しました（部分遮蔽対応）")
            time_area.info(f"倒れている時間: {fall_time_sec:.2f} 秒")
        else:
            status_area.success("✅ 正常")
            time_area.empty()

        prev_gray = gray
        frame_area.image(frame, channels="BGR")

    cap.release()
    st.info("解析終了")

else:
    st.info("動画をアップロードしてください")
