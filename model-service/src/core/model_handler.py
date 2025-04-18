import os
import shutil
from typing import Dict, List, Optional
import requests
from ultralytics import YOLO

class ModelHandler:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.available_models = {
            'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
            'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
            'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
            'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
            'yolov8n-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt'
        }
        
    def download_model(self, model_name: str) -> Dict:
        """Baixa um modelo do repositório oficial."""
        try:
            if model_name not in self.available_models:
                return {
                    'status': 'error',
                    'message': f'Modelo {model_name} não encontrado'
                }
                
            url = self.available_models[model_name]
            save_path = os.path.join(self.model_dir, f'{model_name}.pt')
            
            # Verificar se o modelo já existe
            if os.path.exists(save_path):
                return {
                    'status': 'success',
                    'message': f'Modelo {model_name} já existe'
                }
                
            # Baixar modelo
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
                
            return {
                'status': 'success',
                'message': f'Modelo {model_name} baixado com sucesso'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
            
    def list_models(self) -> List[str]:
        """Lista todos os modelos disponíveis."""
        models = []
        
        # Listar modelos online
        models.extend(list(self.available_models.keys()))
        
        # Listar modelos locais
        for file in os.listdir(self.model_dir):
            if file.endswith('.pt'):
                model_name = file[:-3]  # Remover extensão .pt
                if model_name not in models:
                    models.append(model_name)
                    
        return models
        
    def get_model_info(self, model_name: str) -> Dict:
        """Obtém informações sobre um modelo específico."""
        try:
            model_path = os.path.join(self.model_dir, f'{model_name}.pt')
            
            if not os.path.exists(model_path):
                return {
                    'status': 'error',
                    'message': f'Modelo {model_name} não encontrado localmente'
                }
                
            model = YOLO(model_path)
            
            return {
                'status': 'success',
                'name': model_name,
                'type': model.type,
                'task': model.task,
                'size': os.path.getsize(model_path)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
            
    def delete_model(self, model_name: str) -> Dict:
        """Remove um modelo local."""
        try:
            model_path = os.path.join(self.model_dir, f'{model_name}.pt')
            
            if not os.path.exists(model_path):
                return {
                    'status': 'error',
                    'message': f'Modelo {model_name} não encontrado'
                }
                
            os.remove(model_path)
            
            return {
                'status': 'success',
                'message': f'Modelo {model_name} removido com sucesso'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            } 