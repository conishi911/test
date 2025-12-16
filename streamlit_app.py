import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="転倒検知システム", layout="wide")
st.title("📷 カメラによる転倒検知システム")

run = st.checkbox("カメラ起動")

FRAME_WINDOW = st.image([])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def is_fallen(landmarks, img_h):
    """
    肩と腰のY座標差で転倒判定
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * img_h
    hip_y = (left_hip.y + right_hip.y) / 2 * img_h

    diff = abs(shoulder_y - hip_y)

    # 閾値（ピクセル）
    return diff < 80

status_text = st.empty()

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("カメラを取得できません")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        fallen = is_fallen(results.pose_landmarks.landmark, h)

        if fallen:
            status_text.error("⚠️ 人が倒れています！")
            cv2.putText(frame, "FALL DETECTED", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            status_text.success("✅ 正常姿勢")
            cv2.putText(frame, "NORMAL", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    else:
        status_text.warning("人物が検出されていません")

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
