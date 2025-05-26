import onnxruntime as ort
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple

class YOLOv11Detector:
    def __init__(self, model_path: str = "final_model_yolo11.onnx"):
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", model_path)
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model file not found at {model_path}. "
                f"Please place the file 'final_model_yolo11.onnx' in the 'service/models/' directory."
            )
        self.session = ort.InferenceSession(model_path)
        self.classes = ['chien a pieds', 'chien assis', 'chien au pieds', 'chien debout']
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (640, 640)  # Taille d'entrée du modèle

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, int, int, int, int]:
        # Obtenir les dimensions originales
        orig_h, orig_w = image.shape[:2]
        # Déterminer la dimension minimale pour maintenir les proportions
        scale = min(self.input_shape[0] / orig_h, self.input_shape[1] / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        # Redimensionner avec padding
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.input_shape[0], self.input_shape[1], 3), 128, dtype=np.uint8)  # Gris comme fond
        dw, dh = (self.input_shape[1] - new_w) // 2, (self.input_shape[0] - new_h) // 2
        padded[dh:new_h + dh, dw:new_w + dw, :] = resized
        # Normaliser
        padded = padded.transpose(2, 0, 1)
        padded = padded.astype(np.float32) / 255.0
        padded = np.expand_dims(padded, axis=0)
        return padded, scale, dw, dh, orig_w, orig_h

    def postprocess(self, outputs: np.ndarray, scale: float, dw: float, dh: float, orig_w: int, orig_h: int, confidence_threshold: float = 0.5, iou_threshold: float = 0.5) -> List[Dict]:
        detections = []
        if not outputs or len(outputs) == 0:
            print("No outputs from model")
            return detections

        boxes = outputs[0]  # (1, 8, 8400)
        print(f"Boxes shape: {boxes.shape}")
        print(f"Boxes content: {boxes}")

        if boxes.size == 0 or len(boxes.shape) != 3:
            print("Invalid boxes shape or no detections")
            return detections

        # Transposer les dimensions pour obtenir (1, 8400, 8)
        boxes = boxes.transpose(0, 2, 1)  # Maintenant (1, 8400, 8)
        print(f"Boxes shape after transpose: {boxes.shape}")

        if boxes.shape[2] != (4 + len(self.classes)):
            print(f"Expected {4 + len(self.classes)} values per box, but got {boxes.shape[2]}")
            return detections

        # Itérer sur les boîtes
        for box in boxes[0]:
            try:
                # Coordonnées brutes : [center_x, center_y, width, height]
                center_x, center_y, width, height = box[:4].astype(float)
                scores = box[4:8].astype(float)

                # Vérifier les valeurs finies
                if not (np.isfinite(center_x) and np.isfinite(center_y) and np.isfinite(width) and np.isfinite(height)):
                    continue
                if not np.all(np.isfinite(scores)):
                    continue

                confidence = np.max(scores)
                class_id = np.argmax(scores)

                if confidence > confidence_threshold:
                    # Convertir [center_x, center_y, width, height] en [x1, y1, x2, y2]
                    x1 = max(0, center_x - width / 2)
                    y1 = max(0, center_y - height / 2)
                    x2 = min(self.input_shape[0], center_x + width / 2)
                    y2 = min(self.input_shape[0], center_y + height / 2)

                    # Ajuster dynamiquement la taille de la boîte
                    aspect_ratio = height / width if width > 0 else 1.0
                    max_width = self.input_shape[1] * 0.8  # 80% de la largeur max
                    max_height = self.input_shape[0] * 0.9  # 90% de la hauteur max
                    if width > max_width:
                        width = max_width
                        x1 = max(0, center_x - width / 2)
                        x2 = min(self.input_shape[0], center_x + width / 2)
                    if height > max_height:
                        height = max_height
                        y1 = max(0, center_y - height / 2)
                        y2 = min(self.input_shape[0], center_y + height / 2)
                    # Ajuster la hauteur pour les postures élancées
                    if aspect_ratio > 1.5 and height < max_height:  # Si la posture est élancée
                        height = min(max_height, height * 1.5)  # Augmenter la hauteur avec un facteur 1.5
                        y1 = max(0, center_y - height / 2)
                        y2 = min(self.input_shape[0], center_y + height / 2)

                    print(f"Raw box: [center_x={center_x}, center_y={center_y}, width={width}, height={height}]")
                    print(f"Normalized box (before adjustment): [x1={x1}, y1={y1}, x2={x2}, y2={y2}], confidence={confidence}")

                    # Ajuster les coordonnées en fonction du padding et des dimensions originales
                    x1 = (x1 - dw) / scale
                    y1 = (y1 - dh) / scale
                    x2 = (x2 - dw) / scale
                    y2 = (y2 - dh) / scale

                    # Clipper aux dimensions originales avec un léger relâchement
                    x1 = max(0, min(x1, orig_w * 1.1))  # Permettre un léger dépassement
                    y1 = max(0, min(y1, orig_h * 1.1))
                    x2 = max(0, min(x2, orig_w * 1.1))
                    y2 = max(0, min(y2, orig_h * 1.1))

                    # Réajuster si nécessaire pour rester dans les limites
                    x2 = min(x2, orig_w)
                    y2 = min(y2, orig_h)

                    print(f"Adjusted box: [x1={x1}, y1={y1}, x2={x2}, y2={y2}], confidence={confidence}")

                    if x2 <= x1 or y2 <= y1:
                        continue

                    detections.append({
                        "class_name": self.classes[class_id],
                        "confidence": float(confidence),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
            except Exception as e:
                print(f"Error processing box: {box}, error: {str(e)}")
                continue

        if detections:
            try:
                boxes = np.array([d["bbox"] for d in detections])
                scores = np.array([d["confidence"] for d in detections])
                nms_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    if width > 0 and height > 0:
                        nms_boxes.append([float(x1), float(y1), float(width), float(height)])
                    else:
                        print(f"Invalid box for NMS: {box}")
                        continue

                if nms_boxes:
                    indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), confidence_threshold, iou_threshold)
                    if isinstance(indices, np.ndarray):
                        indices = indices.flatten()
                    elif isinstance(indices, tuple):
                        indices = indices[0] if indices else []
                    detections = [detections[i] for i in indices] if indices else []
                else:
                    print("No valid boxes for NMS")
            except Exception as e:
                print(f"Error in NMS: {str(e)}")
                print("Returning raw detections due to NMS failure")

        print(f"Number of detections after NMS: {len(detections)}")
        return detections

    def process_image(self, image_np: np.ndarray, output_path: str = None) -> List[Dict]:
        # Prétraitement avec conservation des proportions et retour des métadonnées
        input_tensor, scale, dw, dh, orig_w, orig_h = self.preprocess(image_np)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        print(f"Outputs shape: {[o.shape for o in outputs]}")
        detections = self.postprocess(outputs, scale, dw, dh, orig_w, orig_h)

        # Ajuster les annotations pour les dimensions originales
        annotated_image = image_np.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            text_y = max(y1 - 10, 20)
            cv2.putText(annotated_image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if output_path:
            cv2.imwrite(output_path, annotated_image)

        return detections
