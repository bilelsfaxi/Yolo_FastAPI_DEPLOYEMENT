from fastapi import FastAPI
from routers import yolo

app = FastAPI(title="YOLOv11 Dog Posture Detection API")

# Inclure les routes définies dans routers/yolo.py
app.include_router(yolo.router)

@app.get("/")
async def root():
    return {"message": "YOLOv11 Dog Posture Detection API. Use POST /yolo/predict to upload an image."}