import os
import json
import yaml
import shutil
import threading
from datetime import datetime
from PIL import Image
import io
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('treino-yolo')

class YoloTrainer:
    """Classe para gerenciar o treinamento de modelos YOLO para detecção de objetos."""
    
    def __init__(self, base_dir='.'):
        self.base_dir = base_dir
        self.dataset_dir = os.path.join(base_dir, 'dataset_treinamento')
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.labels_dir = os.path.join(self.dataset_dir, 'labels')
        self.config_path = os.path.join(base_dir, 'dataset_config.yaml')
        
        # Carregar classes do arquivo YAML de referência
        yaml_ref_path = r"c:\Users\Diego\Desktop\Projeto de Deteccao do uso de EPI.v3i.yolov8\data.yaml"
        try:
            with open(yaml_ref_path, 'r', encoding='utf-8') as f:
                config_ref = yaml.safe_load(f)
                # Criar dicionário de classes usando índices como chaves
                self.classes = {}
                for idx, name in enumerate(config_ref.get('names', [])):
                    self.classes[idx] = name
                logger.info(f"Classes carregadas do arquivo de referência: {self.classes}")
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo YAML de referência: {str(e)}")
            # Classes padrão caso falhe a leitura
            self.classes = {
                0: 'Pessoa',
                1: 'Capacete de seguranca',
                2: 'Oculos de protecao',
                3: 'Mascara',
                4: 'Luvas de protecao',
                5: 'Roupa de protecao',
                6: 'Botas de seguranca',
                7: 'Abafador de ruido'
            }
        
        self.setup_directories()
        
        # Informações sobre o último treinamento
        self.last_training = {
            'status': None,  # 'running', 'success', 'failed'
            'start_time': None,
            'end_time': None,
            'message': None,
            'model_path': None
        }
    
    def setup_directories(self):
        """Cria a estrutura de diretórios necessária para treinamento."""
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'runs'), exist_ok=True)
        
        logger.info(f"Diretórios criados/verificados:")
        logger.info(f"- Dataset: {self.dataset_dir}")
        logger.info(f"- Imagens: {self.images_dir}")
        logger.info(f"- Labels: {self.labels_dir}")
    
    def save_annotation(self, image_file, annotations):
        """
        Salva uma imagem e suas anotações para o dataset de treinamento.
        
        Args:
            image_file: Arquivo de imagem do request.files
            annotations: String JSON com as anotações das bounding boxes
            
        Returns:
            dict: Informações sobre a operação
        """
        try:
            # Abrir a imagem
            image = Image.open(io.BytesIO(image_file.read()))
            
            # Converter para RGB se necessário
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                image = image.convert('RGB')
            
            # Processar anotações
            try:
                if isinstance(annotations, str):
                    annotations = json.loads(annotations)
            except Exception as e:
                logger.error(f"Erro ao processar JSON de anotações: {e}")
                return {
                    'status': 'error',
                    'message': f'Formato inválido de anotações: {str(e)}'
                }
            
            # Verificar se temos anotações válidas
            if not isinstance(annotations, list) or len(annotations) == 0:
                return {
                    'status': 'error',
                    'message': 'Nenhuma anotação encontrada ou formato inválido'
                }
            
            # Criar nome de arquivo com timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            img_filename = f"{timestamp}.jpg"
            txt_filename = f"{timestamp}.txt"
            
            # Caminhos completos para os arquivos
            image_path = os.path.join(self.images_dir, img_filename)
            label_path = os.path.join(self.labels_dir, txt_filename)
            
            # Salvar a imagem
            image.save(image_path)
            logger.info(f"Imagem salva em: {image_path}")
            
            # Salvar as anotações no formato YOLO
            with open(label_path, 'w') as f:
                for annotation in annotations:
                    class_id = annotation.get('class_id', 0)
                    x = annotation.get('x', 0)
                    y = annotation.get('y', 0)
                    w = annotation.get('width', 0)
                    h = annotation.get('height', 0)
                    
                    # Escrever no formato YOLO
                    f.write(f"{class_id} {x} {y} {w} {h}\n")
            
            logger.info(f"Anotações salvas em: {label_path}")
            
            # Retornar informações
            num_images = len(os.listdir(self.images_dir))
            return {
                'status': 'success',
                'message': 'Imagem e anotações salvas com sucesso',
                'filename': img_filename,
                'num_images': num_images
            }
            
        except Exception as e:
            logger.error(f"Erro ao salvar anotação: {str(e)}")
            return {
                'status': 'error',
                'message': f'Erro ao salvar anotação: {str(e)}'
            }
    
    def generate_config(self):
        """Gera o arquivo de configuração YAML para treinamento."""
        try:
            # Definir configuração
            config = {
                'path': self.dataset_dir,  # Diretório base do dataset
                'train': './images',       # Caminho relativo para imagens de treino
                'val': './images',         # Caminho relativo para imagens de validação
                'nc': len(self.classes),   # Número de classes
                'names': list(self.classes.values())  # Lista de nomes das classes na ordem correta
            }
            
            # Salvar configuração
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, sort_keys=False)
            
            logger.info(f"Arquivo de configuração criado em: {self.config_path}")
            logger.info(f"Classes configuradas: {config['names']}")
            return True
        except Exception as e:
            logger.error(f"Erro ao gerar arquivo de configuração: {str(e)}")
            return False
    
    def start_training(self, epochs=100, batch_size=16, img_size=640, patience=20):
        """
        Inicia o treinamento do modelo YOLOv8 com o dataset atual.
        
        Returns:
            dict: Informações sobre a operação
        """
        # Verificar número de imagens
        num_images = len(os.listdir(self.images_dir))
        if num_images < 10:
            return {
                'status': 'error',
                'message': f'Você precisa de pelo menos 10 imagens para treinar. Você tem apenas {num_images}.'
            }
        
        # Gerar arquivo de configuração
        if not self.generate_config():
            return {
                'status': 'error',
                'message': 'Erro ao gerar arquivo de configuração para treinamento.'
            }
        
        # Iniciar treinamento em thread separada
        def train_thread():
            try:
                self.last_training = {
                    'status': 'running',
                    'start_time': datetime.now(),
                    'end_time': None,
                    'message': 'Treinamento em andamento...',
                    'model_path': None
                }
                
                logger.info("Iniciando treinamento do modelo...")
                from ultralytics import YOLO
                
                # Carregar modelo base
                model = YOLO('yolov8n.pt')
                
                # Iniciar treinamento
                results = model.train(
                    data=self.config_path,
                    epochs=epochs,
                    imgsz=img_size,
                    batch=batch_size,
                    name='epi_detector',
                    patience=patience,
                    save=True
                )
                
                # Treinamento concluído
                best_model_path = os.path.join('runs', 'detect', 'epi_detector', 'weights', 'best.pt')
                
                if os.path.exists(best_model_path):
                    # Copiar para a raiz para facilitar o acesso
                    shutil.copy(best_model_path, os.path.join(self.base_dir, 'epi_detector.pt'))
                    
                    self.last_training = {
                        'status': 'success',
                        'start_time': self.last_training['start_time'],
                        'end_time': datetime.now(),
                        'message': 'Treinamento concluído com sucesso!',
                        'model_path': best_model_path
                    }
                    
                    logger.info("Treinamento concluído com sucesso!")
                    logger.info(f"Modelo salvo em: {os.path.abspath(best_model_path)}")
                else:
                    self.last_training = {
                        'status': 'failed',
                        'start_time': self.last_training['start_time'],
                        'end_time': datetime.now(),
                        'message': 'Treinamento concluído, mas o modelo não foi encontrado.',
                        'model_path': None
                    }
                    
                    logger.error("Treinamento concluído, mas o modelo não foi encontrado.")
                
            except Exception as e:
                self.last_training = {
                    'status': 'failed',
                    'start_time': self.last_training['start_time'],
                    'end_time': datetime.now(),
                    'message': f'Erro durante treinamento: {str(e)}',
                    'model_path': None
                }
                
                logger.error(f"Erro durante treinamento: {str(e)}")
        
        # Iniciar a thread de treinamento
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
        
        return {
            'status': 'success',
            'message': 'Treinamento iniciado em segundo plano. Este processo pode demorar várias horas.',
            'num_images': num_images
        }
    
    def get_training_status(self):
        """Retorna o status atual do treinamento."""
        if self.last_training['status'] == 'running':
            # Calcular tempo decorrido
            elapsed = datetime.now() - self.last_training['start_time']
            minutes = int(elapsed.total_seconds() / 60)
            
            # Adicionar informação de tempo ao status
            self.last_training['elapsed_minutes'] = minutes
        
        return self.last_training
    
    def get_dataset_info(self):
        """Retorna informações sobre o dataset atual."""
        num_images = len(os.listdir(self.images_dir))
        num_labels = len(os.listdir(self.labels_dir))
        
        return {
            'num_images': num_images,
            'num_labels': num_labels,
            'classes': self.classes,
            'dataset_path': self.dataset_dir,
            'ready_for_training': num_images >= 10
        }
    
    def clear_dataset(self):
        """Limpa o dataset atual (remove todas as imagens e anotações)."""
        try:
            # Remover arquivos de imagens
            for file in os.listdir(self.images_dir):
                file_path = os.path.join(self.images_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Remover arquivos de anotações
            for file in os.listdir(self.labels_dir):
                file_path = os.path.join(self.labels_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            logger.info("Dataset limpo com sucesso.")
            return {
                'status': 'success',
                'message': 'Dataset limpo com sucesso.'
            }
        except Exception as e:
            logger.error(f"Erro ao limpar dataset: {str(e)}")
            return {
                'status': 'error',
                'message': f'Erro ao limpar dataset: {str(e)}'
            }

# Inicializar o treinador
trainer = YoloTrainer() 