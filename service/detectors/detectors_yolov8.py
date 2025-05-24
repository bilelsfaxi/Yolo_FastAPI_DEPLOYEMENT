import onnxruntime as ort
import os
import gdown
import cv2
import numpy as np
from typing import List, Dict

class YOLOv11Detector:
    def __init__(self, model_path: str = "final_model_yolo11.onnx"):
        # Déterminer le chemin absolu du modèle
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", model_path)
        
        # Vérifier si le modèle existe localement, sinon le télécharger depuis Google Drive
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Téléchargement du modèle depuis Google Drive vers {model_path}...")
            gdown.download("https://drive.google.com/uc?id=VOTRE_FILE_ID", model_path, quiet=False)
        
        # Charger le modèle ONNX
        self.session = ort.InferenceSession(model_path)
        self.classes = ["au pieds", "à pieds", "debout", "assis"]
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (640, 640) # Taille d'entrée (ajustez si différente)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Redimensionner l'image
        img = cv2.resize(image, self.input_shape)
        # Convertir en format [1, C, H, W]
        img = img.transpose(2, 0, 1)  # De HWC à CHW
        img = img.astype(np.float32) / 255.0  # Normalisation
        img = np.expand_dims(img, axis=0)  # Ajouter dimension batch
        return img

    def postprocess(self, outputs: np.ndarray, confidence_threshold: float = 0.5, iou_threshold: float = 0.5) -> List[Dict]:
        # Supposer que l'output est de forme [1, num_boxes, 4 + num_classes] (bbox + scores)
        detections = []
        boxes = outputs[0]  # [num_boxes, 4 + num_classes]

        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            scores = box[4:]
            confidence = np.max(scores)
            class_id = np.argmax(scores)

            if confidence > confidence_threshold:
                detections.append({
                    "class_name": self.classes[class_id],
                    "confidence": float(confidence),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

        # Appliquer NMS (Non-Maximum Suppression)
        if detections:
            boxes = np.array([d["bbox"] for d in detections])
            scores = np.array([d["confidence"] for d in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), confidence_threshold, iou_threshold)
            detections = [detections[i] for i in indices]

        return detections

    def process_image(self, input_path: str, output_path: str) -> List[Dict]:
        # Charger l'image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("Could not load image")

        # Prétraitement
        input_tensor = self.preprocess(image)

        # Exécuter l'inférence
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Post-traitement
        detections = self.postprocess(outputs)

        # Annoter l'image
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            confidence = detection["confidence"]
            class_name = detection["class_name"]

            # Dessiner la boîte englobante et le label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sauvegarder l'image annotée
        cv2.imwrite(output_path, image)

        return detections