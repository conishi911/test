import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="転倒検知", layout="wide")
st.title("📷 人体転倒検知システム（MediaPipe不要）")

run = st.checkbox("カメラ起動")
FRAME_WINDOW = st.image([])

# 人検出器（HOG）
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

status = st.empty()

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("カメラを取得できません")
        break

    frame = cv2.flip(frame, 1)

    boxes, _ = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    fallen_detected = False

    for (x, y, w, h) in boxes:
        aspect_ratio = w / h

        if aspect_ratio > 1.2:
            fallen_detected = True
            color = (0, 0, 255)
            label = "FALL DETECTED"
        else:
            color = (0, 255, 0)
            label = "NORMAL"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if fallen_detected:
        status.error("⚠️ 人が倒れています")
    else:
        status.success("✅ 正常姿勢")

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
