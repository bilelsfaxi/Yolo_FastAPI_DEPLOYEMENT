from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from service.schemas import DetectionResponse
from service.detectors import YOLOv11Detector
import os
import tempfile

router = APIRouter(prefix="/yolo", tags=["YOLOv11"])

# Initialiser le détecteur avec le chemin du modèle ONNX
detector = YOLOv11Detector(model_path="C:\Users\GIGABYTE\Downloads\API_deployment\service\final_model_yolo11.onnx")

@router.post("/predict", response_model=DetectionResponse)
async def predict(file: UploadFile = File(...)):
    # Valider le type de fichier
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Créer un fichier temporaire pour sauvegarder l'image d'entrée
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_input:
            temp_input.write(await file.read())
            temp_input_path = temp_input.name

        # Générer un chemin pour l'image annotée
        temp_output_path = temp_input_path.replace(".jpg", "_annotated.jpg")

        # Traiter l'image avec YOLOv11
        detections = detector.process_image(temp_input_path, temp_output_path)

        # Vérifier si l'image annotée a été créée
        if not os.path.exists(temp_output_path):
            raise HTTPException(status_code=500, detail="Failed to generate annotated image")

        # Renvoyer l'image annotée comme fichier téléchargeable
        return FileResponse(
            temp_output_path,
            media_type="image/jpeg",
            filename="annotated_image.jpg",
            headers={"X-Detections": str(detections)}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    finally:
        # Nettoyer les fichiers temporaires
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)