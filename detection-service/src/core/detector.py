from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from ultralytics import YOLO

class EPIDetector:
    def __init__(self):
        self.model = None
        self.current_model = None
        self.available_models = {
            'yolov8n': 'yolov8n.pt',
            'yolov8s': 'yolov8s.pt',
            'yolov8m': 'yolov8m.pt',
            'yolov8l': 'yolov8l.pt',
            'yolov8x': 'yolov8x.pt',
            'yolov8n-pose': 'yolov8n-pose.pt',
            'local_default': 'model.pt',
            'epi_detector': 'runs/detect/epi_detector/weights/best.pt',
            'epi_detector2': 'runs/detect/epi_detector2/weights/best.pt'
        }
        
    def load_model(self, model_name: str) -> bool:
        """Carrega um modelo específico."""
        try:
            if model_name not in self.available_models:
                raise ValueError(f"Modelo {model_name} não encontrado")
                
            model_path = self.available_models[model_name]
            self.model = YOLO(model_path)
            self.current_model = model_name
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo: {str(e)}")
            return False
            
    def detect_epis(self, image: np.ndarray, selected_epis: List[str]) -> Dict[str, Any]:
        """Detecta EPIs em uma imagem."""
        if self.model is None:
            raise ValueError("Nenhum modelo carregado")
            
        try:
            # Realizar detecção
            results = self.model(image)[0]
            detections = []
            
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result
                class_name = results.names[int(cls)]
                
                if class_name in selected_epis:
                    detections.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'box': [float(x1), float(y1), float(x2), float(y2)]
                    })
            
            # Verificar EPIs faltantes
            missing_epis = [epi for epi in selected_epis 
                           if not any(d['class'] == epi for d in detections)]
            
            return {
                'status': 'success',
                'detections': detections,
                'missing_epis': missing_epis,
                'model_used': self.current_model
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def detect_specific_epi(self, image: np.ndarray, epi_type: str) -> Dict[str, Any]:
        """Detecta um EPI específico com maior precisão."""
        try:
            results = self.model(image)[0]
            
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = result
                class_name = results.names[int(cls)]
                
                if class_name == epi_type:
                    return {
                        'status': 'success',
                        'detected': True,
                        'confidence': float(conf),
                        'coordinates': [float(x1), float(y1), float(x2), float(y2)]
                    }
            
            return {
                'status': 'success',
                'detected': False
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            } 