from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from ultralytics import YOLO
import os
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.join("model", "yolov11.pt")
model = YOLO(model_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    file_bytes = np.asarray(bytearray(contents), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    results = model(image)
    detection_labels = []
    color = (50, 168, 129)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = model.names

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[class_id]}: {round(score*100)}%"
            detection_labels.append(label)

            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_width, text_height = text_size
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(image, (x1, y2 - text_height - 5), (x1 + text_width, y2), color, -1)
            cv2.putText(image, label, (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={
        "labels": detection_labels,
        "image_base64": image_base64
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
