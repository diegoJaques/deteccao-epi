import os
import sys
import time
import json
import logging
import requests
import traceback
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from dotenv import load_dotenv
import io
import base64
import yaml
import cv2

# Importar o módulo de treinamento
from treinamento import trainer

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('epi-detector')

# Carregar variáveis de ambiente
load_dotenv()

# Inicializar a aplicação Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'chave_secreta_epi_detector')

# Configurações
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'resultados'
MODELS_FOLDER = 'models'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Configurar tratamento de erro para arquivos muito grandes
@app.errorhandler(413)
def request_entity_too_large(error):
    tamanho_max_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
    return jsonify({
        "status": "erro",
        "mensagem": f"A imagem é muito grande. O tamanho máximo permitido é {tamanho_max_mb}MB."
    }), 413

# Criar diretórios necessários
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Variáveis globais
modelo_atual = None
nome_modelo_atual = None

# Função para verificar extensões permitidas
def extensao_permitida(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Carregar modelo YOLO
def carregar_modelo(nome_modelo=None):
    """
    Carrega um modelo YOLO do diretório de modelos.
    
    Args:
        nome_modelo: Nome do arquivo do modelo (ex: 'yolov8n.pt')
    
    Returns:
        O modelo YOLO carregado ou None se ocorrer erro
    """
    global modelo_atual, nome_modelo_atual
    
    try:
        # Se nenhum modelo for especificado, procura um modelo por padrão
        if nome_modelo is None:
            modelos_disponiveis = os.listdir(MODELS_FOLDER)
            modelos_pt = [m for m in modelos_disponiveis if m.endswith('.pt')]
            
            if not modelos_pt:
                # Se não houver modelos locais, baixa o YOLOv8n
                logger.info("Nenhum modelo encontrado. Baixando YOLOv8n...")
                modelo = YOLO('yolov8n.pt')
                modelo.save(os.path.join(MODELS_FOLDER, 'yolov8n.pt'))
                nome_modelo = 'yolov8n.pt'
            else:
                # Prioriza modelos específicos para EPI
                for candidato in ['epi_detector.pt', 'yolov8n.pt', 'yolov8s.pt']:
                    if candidato in modelos_pt:
                        nome_modelo = candidato
                        break
                if nome_modelo is None:
                    nome_modelo = modelos_pt[0]  # usa o primeiro modelo disponível
        
        # Caminho completo para o modelo
        caminho_modelo = os.path.join(MODELS_FOLDER, nome_modelo)
        
        # Verifica se o arquivo existe
        if not os.path.exists(caminho_modelo):
            logger.error(f"Modelo {nome_modelo} não encontrado em {caminho_modelo}")
            # Tenta baixar o modelo da Ultralytics
            modelo = YOLO(nome_modelo)
            modelo.save(caminho_modelo)
            logger.info(f"Modelo {nome_modelo} baixado e salvo em {caminho_modelo}")
        else:
            # Carrega o modelo do arquivo local
            logger.info(f"Carregando modelo de {caminho_modelo}")
            modelo = YOLO(caminho_modelo)
        
        # Atualiza as variáveis globais
        modelo_atual = modelo
        nome_modelo_atual = nome_modelo
        
        logger.info(f"Modelo {nome_modelo} carregado com sucesso!")
        return modelo
    
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        traceback.print_exc()
        return None

# Função para fazer detecção em uma imagem
def detectar_objetos(imagem_path, conf=0.25):
    try:
        # Obter o mapeamento de classes diretamente do modelo carregado
        if modelo_atual is None:
            logger.error("Erro Crítico: Tentando detectar sem modelo carregado.")
            raise Exception("Nenhum modelo YOLO está carregado no momento.")
        
        # Verificar se o modelo carregado possui o atributo 'names'
        if not hasattr(modelo_atual, 'names') or not modelo_atual.names:
            logger.error(f"Erro: Modelo '{nome_modelo_atual}' não possui mapeamento de classes (.names).")
            # Como fallback, tentar usar um mapeamento genérico (mas isso não é ideal)
            class_mapping = {i: f"Classe {i}" for i in range(80)} # Supõe 80 classes (COCO)
        else:
             class_mapping = modelo_atual.names
             
        logger.info(f"Usando mapeamento de classes do modelo '{nome_modelo_atual}': {class_mapping}")
        
        # Carrega a imagem
        img = cv2.imread(imagem_path)
        if img is None:
            raise Exception("Erro ao carregar a imagem")

        # Realiza a detecção
        results = modelo_atual(img)[0]
        
        # Processa os resultados
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = result
            
            # Converte o ID da classe para inteiro
            class_id = int(class_id)
            
            # Obtém o nome da classe do mapeamento - USANDO O INTEIRO class_id COMO CHAVE
            class_name = class_mapping.get(class_id, f"Classe Desconhecida {class_id}") # Nome padrão mais claro
            
            # Log para debug
            logger.info(f"Detectado: ID da classe = {class_id}, Tipo do ID = {type(class_id)}")
            logger.info(f"Procurando por chave: {class_id} (tipo: {type(class_id)}) no mapeamento.")
            logger.info(f"Nome da classe encontrado: {class_name}")
            
            # Adiciona a detecção à lista
            detections.append({
                'class_name': class_name,  # Nome da classe
                'confidence': round(conf * 100, 2),
                'bbox': [round(x) for x in [x1, y1, x2, y2]]
            })
            
            # Desenha a caixa delimitadora
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Adiciona o rótulo
            label = f"{class_name} {conf:.2%}"
            cv2.putText(img, label, (int(x1), int(y1 - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Salva a imagem processada
        output_path = os.path.join(RESULTS_FOLDER, os.path.basename(imagem_path))
        cv2.imwrite(output_path, img)
        
        return {
            'detections': detections,
            'output_path': output_path
        }
        
    except Exception as e:
        logging.error(f"Erro durante a detecção: {str(e)}")
        traceback.print_exc()
        raise

# Rotas da API
@app.route('/')
def index():
    """Página inicial da aplicação"""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def arquivo_upload(filename):
    """Serve arquivos da pasta uploads"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/resultados/<filename>')
def arquivo_resultado(filename):
    """Serve arquivos da pasta resultados"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/modelos', methods=['GET'])
def listar_modelos():
    """Lista todos os modelos disponíveis"""
    try:
        modelos = []
        
        # Listar modelos locais
        for arquivo in os.listdir(MODELS_FOLDER):
            if arquivo.endswith('.pt'):
                caminho = os.path.join(MODELS_FOLDER, arquivo)
                tamanho = os.path.getsize(caminho) / (1024 * 1024)  # em MB
                modelos.append({
                    "nome": arquivo,
                    "local": True,
                    "tamanho_mb": round(tamanho, 2),
                    "caminho": caminho,
                    "atual": arquivo == nome_modelo_atual
                })
        
        # Lista modelos padrão da Ultralytics (não precisa baixar)
        modelos_ultralytics = [
            {"nome": "yolov8n.pt", "descricao": "YOLOv8 Nano"},
            {"nome": "yolov8s.pt", "descricao": "YOLOv8 Small"},
            {"nome": "yolov8m.pt", "descricao": "YOLOv8 Medium"},
            {"nome": "yolov8l.pt", "descricao": "YOLOv8 Large"},
            {"nome": "yolov8x.pt", "descricao": "YOLOv8 Extra Large"}
        ]
        
        # Adiciona modelos da Ultralytics que não estão na lista local
        nomes_locais = [m["nome"] for m in modelos]
        for modelo in modelos_ultralytics:
            if modelo["nome"] not in nomes_locais:
                modelos.append({
                    "nome": modelo["nome"],
                    "local": False,
                    "descricao": modelo["descricao"],
                    "disponivel_download": True,
                    "atual": modelo["nome"] == nome_modelo_atual
                })
        
        return jsonify({
            "status": "sucesso",
            "modelo_atual": nome_modelo_atual,
            "modelos": modelos
        })
    
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {str(e)}")
        return jsonify({
            "status": "erro",
            "mensagem": str(e)
        }), 500

@app.route('/definir_modelo', methods=['POST'])
def definir_modelo():
    """Define o modelo a ser usado para detecção"""
    try:
        # Obter dados do request
        dados = request.get_json()
        
        if not dados or 'modelo' not in dados:
            logger.warning("Requisição inválida: 'modelo' não especificado")
            return jsonify({
                "status": "erro",
                "mensagem": "É necessário especificar um modelo"
            }), 400
        
        nome_modelo = dados['modelo']
        logger.info(f"Solicitação para definir modelo: {nome_modelo}")
        
        # Verificar se o modelo existe antes de tentar carregá-lo
        caminho_modelo = os.path.join(MODELS_FOLDER, nome_modelo)
        if not os.path.exists(caminho_modelo) and not nome_modelo in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
            logger.warning(f"Modelo solicitado não existe: {nome_modelo}")
            return jsonify({
                "status": "erro",
                "mensagem": f"Modelo {nome_modelo} não encontrado"
            }), 404
        
        # Carregar o modelo
        logger.info(f"Tentando carregar o modelo: {nome_modelo}")
        modelo = carregar_modelo(nome_modelo)
        
        if modelo is None:
            logger.error(f"Falha ao carregar o modelo: {nome_modelo}")
            return jsonify({
                "status": "erro",
                "mensagem": f"Não foi possível carregar o modelo {nome_modelo}"
            }), 500
        
        logger.info(f"Modelo carregado com sucesso: {nome_modelo}")
        return jsonify({
            "status": "sucesso",
            "mensagem": f"Modelo {nome_modelo} carregado com sucesso",
            "modelo": nome_modelo
        })
    
    except Exception as e:
        logger.error(f"Erro ao definir modelo: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "erro",
            "mensagem": str(e)
        }), 500

@app.route('/detectar', methods=['POST'])
def detectar():
    try:
        # Verifica se há arquivo na requisição
        if 'imagem' not in request.files:
            return jsonify({"erro": "Nenhum arquivo enviado"}), 400
            
        arquivo = request.files['imagem']
        if arquivo.filename == '':
            return jsonify({"erro": "Nome do arquivo vazio"}), 400
            
        # Verifica a extensão do arquivo
        if not arquivo.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"erro": "Formato de arquivo não suportado"}), 400
            
        # Verifica o tamanho do arquivo (limite de 10MB)
        if len(arquivo.read()) > 10 * 1024 * 1024:  # 10MB em bytes
            return jsonify({"erro": "Arquivo muito grande (máximo 10MB)"}), 400
        arquivo.seek(0)  # Reset do ponteiro do arquivo
        
        # Obtém o nível de confiança da requisição (padrão 0.25)
        conf = float(request.form.get('conf', 0.25))
        if not 0 <= conf <= 1:
            return jsonify({"erro": "Nível de confiança deve estar entre 0 e 1"}), 400
            
        # Salva o arquivo com timestamp para evitar conflitos
        timestamp = int(time.time())
        nome_arquivo = f"upload_{timestamp}_{secure_filename(arquivo.filename)}"
        caminho_arquivo = os.path.join(UPLOAD_FOLDER, nome_arquivo)
        arquivo.save(caminho_arquivo)
        
        # Realiza a detecção
        inicio = time.time()
        resultados = detectar_objetos(caminho_arquivo, conf)
        tempo_deteccao = time.time() - inicio
        
        # Log detalhado das detecções ANTES de enviar
        logger.info(f"Detecções processadas pela função detectar_objetos: {resultados['detections']}")
        
        # Prepara a resposta
        resposta = {
            "status": "sucesso",
            "url_imagem_original": f"/uploads/{nome_arquivo}",
            "url_imagem_processada": f"/resultados/{os.path.basename(resultados['output_path'])}",
            "tempo_deteccao": round(tempo_deteccao, 2),
            "total_deteccoes": len(resultados['detections']),
            "deteccoes": resultados['detections']
        }
        
        return jsonify(resposta)
        
    except Exception as e:
        logger.error(f"Erro durante a detecção: {str(e)}")
        return jsonify({"erro": str(e)}), 500

@app.route('/info', methods=['GET'])
def info_sistema():
    """Retorna informações sobre o sistema"""
    try:
        # Informações do hardware
        gpu_disponivel = torch.cuda.is_available()
        if gpu_disponivel:
            dispositivo = f"GPU: {torch.cuda.get_device_name(0)}"
            memoria_gpu = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        else:
            dispositivo = "CPU"
            memoria_gpu = 0
        
        # Informações do modelo atual
        info_modelo = {
            "nome": nome_modelo_atual,
            "carregado": modelo_atual is not None
        }
        
        # Status do sistema
        info = {
            "status": "online",
            "hardware": {
                "dispositivo": dispositivo,
                "gpu_disponivel": gpu_disponivel,
                "memoria_gpu_gb": memoria_gpu,
                "cpu_threads": os.cpu_count()
            },
            "modelo": info_modelo,
            "sistema": {
                "versao_python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "versao_torch": torch.__version__,
                "diretorio_trabalho": os.getcwd(),
                "arquivos_upload": len(os.listdir(UPLOAD_FOLDER)),
                "arquivos_resultados": len(os.listdir(RESULTS_FOLDER))
            }
        }
        
        return jsonify(info)
    
    except Exception as e:
        logger.error(f"Erro ao buscar informações: {str(e)}")
        return jsonify({
            "status": "erro",
            "mensagem": str(e)
        }), 500

@app.route('/ping', methods=['GET'])
def ping():
    """Endpoint para verificar se o servidor está online"""
    try:
        return jsonify({
            "status": "success",
            "message": "Server is running",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Erro no endpoint ping: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Rotas para treinamento
@app.route('/treinar', methods=['GET'])
def pagina_treinamento():
    """Página para treinamento de modelos"""
    dataset_info = trainer.get_dataset_info()
    return render_template('treinamento.html', num_imagens=dataset_info['num_images'])

@app.route('/salvar_anotacao', methods=['POST'])
def salvar_anotacao():
    """Salva uma imagem e suas anotações para o dataset de treinamento"""
    try:
        # Verificar se recebemos a imagem
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'Nenhuma imagem enviada'}), 400
        
        # Obter imagem e anotações
        image_file = request.files['image']
        annotations = request.form.get('annotations', '[]')
        
        # Salvar usando o treinador
        result = trainer.save_annotation(image_file, annotations)
        
        if result['status'] == 'error':
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erro ao salvar anotação: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro ao salvar anotação: {str(e)}'
        }), 500

@app.route('/iniciar_treinamento', methods=['POST'])
def iniciar_treinamento():
    """Inicia o treinamento do modelo YOLO com o dataset atual"""
    try:
        # Obter parâmetros (opcionais)
        data = request.get_json() or {}
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 16)
        img_size = data.get('img_size', 640)
        patience = data.get('patience', 20)
        
        # Iniciar treinamento
        result = trainer.start_training(
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            patience=patience
        )
        
        if result['status'] == 'error':
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erro ao iniciar treinamento: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro ao iniciar treinamento: {str(e)}'
        }), 500

@app.route('/status_treinamento', methods=['GET'])
def status_treinamento():
    """Retorna o status atual do treinamento"""
    try:
        status = trainer.get_training_status()
        return jsonify({
            'status': 'success',
            'training_status': status
        })
    except Exception as e:
        logger.error(f"Erro ao obter status do treinamento: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro ao obter status do treinamento: {str(e)}'
        }), 500

@app.route('/info_dataset', methods=['GET'])
def info_dataset():
    """Retorna informações sobre o dataset atual"""
    try:
        info = trainer.get_dataset_info()
        return jsonify({
            'status': 'success',
            'dataset_info': info
        })
    except Exception as e:
        logger.error(f"Erro ao obter informações do dataset: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro ao obter informações do dataset: {str(e)}'
        }), 500

@app.route('/limpar_dataset', methods=['POST'])
def limpar_dataset():
    """Limpa o dataset atual (remove todas as imagens e anotações)"""
    try:
        result = trainer.clear_dataset()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erro ao limpar dataset: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro ao limpar dataset: {str(e)}'
        }), 500

@app.route('/api/dataset/clear', methods=['POST'])
def api_clear_dataset():
    """Alias para /limpar_dataset para compatibilidade com a interface do treinar.html"""
    return limpar_dataset()

@app.route('/api/dataset/status', methods=['GET'])
def api_dataset_status():
    """Retorna informações sobre o status atual do dataset de treinamento."""
    try:
        info = trainer.get_dataset_info()
        return jsonify({
            'success': True,
            'total_images': info.get('num_images', 0),
            'annotated_images': info.get('num_labels', 0),
            'class_counts': {},  # Implementar contagem por classe se necessário
            'ready_for_training': info.get('ready_for_training', False)
        })
    except Exception as e:
        logger.error(f"Erro ao obter status do dataset: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dataset/images', methods=['GET'])
def api_dataset_images():
    """Retorna a lista de imagens no dataset."""
    try:
        # Obter caminhos dos diretórios
        images_dir = os.path.join('dataset_treinamento', 'images')
        labels_dir = os.path.join('dataset_treinamento', 'labels')
        
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
        
        images = []
        for idx, img_file in enumerate(os.listdir(images_dir)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(images_dir, img_file)
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)
                
                # Verificar se tem anotações
                has_annotations = os.path.exists(label_path)
                annotations_count = 0
                
                if has_annotations:
                    try:
                        with open(label_path, 'r') as f:
                            annotations_count = len(f.readlines())
                    except:
                        pass
                
                # Criar thumbnail base64 da imagem
                try:
                    img = Image.open(image_path)
                    img.thumbnail((100, 100))
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    thumbnail = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
                except Exception as e:
                    thumbnail = ""
                    logger.error(f"Erro ao criar thumbnail para {img_file}: {str(e)}")
                
                images.append({
                    'id': idx,
                    'filename': img_file,
                    'has_annotations': has_annotations,
                    'annotations_count': annotations_count,
                    'thumbnail': thumbnail
                })
        
        return jsonify({
            'success': True,
            'images': images
        })
    except Exception as e:
        logger.error(f"Erro ao obter imagens do dataset: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/classes', methods=['GET'])
def api_classes():
    """Retorna a lista de classes disponíveis para anotação."""
    try:
        # Carregar classes do arquivo de configuração YAML
        classes = []
        yaml_path = 'dataset_config.yaml'
        
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'names' in config:
                    classes = list(config['names'].values())
        
        # Se não encontrar classes no arquivo, usar as classes do treinador
        if not classes:
            info = trainer.get_dataset_info()
            classes = info.get('classes', {}).values()
        
        # Se ainda não tiver classes, usar padrão
        if not classes:
            classes = ['pessoa', 'capacete', 'oculos_protecao', 'mascara', 'luvas', 'colete']
        
        return jsonify({
            'success': True,
            'classes': classes
        })
    except Exception as e:
        logger.error(f"Erro ao obter classes disponíveis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dataset/image/<int:image_id>', methods=['GET'])
def api_get_dataset_image(image_id):
    """Obtém uma imagem específica do dataset com suas anotações."""
    try:
        # Obter lista de imagens
        images_dir = os.path.join('dataset_treinamento', 'images')
        labels_dir = os.path.join('dataset_treinamento', 'labels')
        
        if not os.path.exists(images_dir):
            return jsonify({
                'success': False,
                'error': 'Diretório de imagens não encontrado'
            }), 404
        
        # Listar arquivos de imagem
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Verificar se o índice é válido
        if image_id < 0 or image_id >= len(image_files):
            return jsonify({
                'success': False,
                'error': 'ID de imagem inválido'
            }), 404
        
        # Obter nome do arquivo
        filename = image_files[image_id]
        image_path = os.path.join(images_dir, filename)
        
        # Carregar imagem e converter para base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Verificar se existem anotações
        annotations = []
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        if os.path.exists(label_path):
            # Abrir imagem para obter dimensões
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Ler arquivo de anotações
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        
                        # Obter nome da classe
                        class_name = None
                        info = trainer.get_dataset_info()
                        classes = info.get('classes', {})
                        if class_id in classes:
                            class_name = classes[class_id]
                        else:
                            class_name = f"classe_{class_id}"
                        
                        # Converter de formato YOLO para pixels
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Converter para coordenadas de pixels
                        x = int((x_center - width/2) * img_width)
                        y = int((y_center - height/2) * img_height)
                        w = int(width * img_width)
                        h = int(height * img_height)
                        
                        annotations.append({
                            'class': class_name,
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h
                        })
        
        return jsonify({
            'success': True,
            'filename': filename,
            'image_data': f'data:image/jpeg;base64,{image_data}',
            'annotations': annotations
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter imagem do dataset: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/annotations/save', methods=['POST'])
def api_save_annotations():
    """Salva anotações de uma imagem no dataset."""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Dados não fornecidos'
            }), 400
            
        # Obter informações enviadas
        filename = data.get('filename')
        image_data = data.get('image_data')
        image_width = data.get('image_width')
        image_height = data.get('image_height')
        annotations = data.get('annotations', [])
        
        if not filename or not annotations:
            return jsonify({
                'success': False,
                'error': 'Dados incompletos'
            }), 400
        
        # Verificar se a imagem foi enviada como base64 ou é um nome de arquivo existente
        if image_data and image_data.startswith('data:image'):
            # Salvar imagem do base64
            try:
                # Extrair dados base64
                img_format, img_str = image_data.split(';base64,')
                img_data = base64.b64decode(img_str)
                
                # Criar diretórios se não existirem
                os.makedirs('dataset_treinamento/images', exist_ok=True)
                os.makedirs('dataset_treinamento/labels', exist_ok=True)
                
                # Salvar imagem
                image_path = os.path.join('dataset_treinamento/images', filename)
                with open(image_path, 'wb') as f:
                    f.write(img_data)
            except Exception as e:
                logger.error(f"Erro ao salvar imagem: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Erro ao salvar imagem: {str(e)}'
                }), 500
        
        # Converter anotações para formato YOLO
        try:
            # Criar arquivo de anotação no formato YOLO
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join('dataset_treinamento/labels', label_filename)
            
            with open(label_path, 'w') as f:
                for annotation in annotations:
                    # Obter índice da classe do arquivo de configuração
                    class_name = annotation.get('class')
                    class_index = -1
                    
                    # Carregar mapeamento de classes
                    yaml_path = 'dataset_config.yaml'
                    if os.path.exists(yaml_path):
                        with open(yaml_path, 'r') as yaml_file:
                            config = yaml.safe_load(yaml_file)
                            if config and 'names' in config:
                                for idx, name in config['names'].items():
                                    if name == class_name:
                                        class_index = int(idx)
                                        break
                    
                    # Se não encontrou, usar 0 (pessoa por padrão)
                    if class_index == -1:
                        logger.warning(f"Classe '{class_name}' não encontrada no config, usando 0")
                        class_index = 0
                    
                    # Coordenadas em formato YOLO (x_center, y_center, width, height)
                    x = annotation.get('x')
                    y = annotation.get('y')
                    width = annotation.get('width')
                    height = annotation.get('height')
                    
                    # Normalizar para valores entre 0 e 1
                    x_center = (x + width / 2) / image_width
                    y_center = (y + height / 2) / image_height
                    norm_width = width / image_width
                    norm_height = height / image_height
                    
                    # Escrever linha no formato YOLO: class_idx x_center y_center width height
                    f.write(f"{class_index} {x_center} {y_center} {norm_width} {norm_height}\n")
        except Exception as e:
            logger.error(f"Erro ao salvar anotações: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Erro ao salvar anotações: {str(e)}'
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Anotações salvas com sucesso'
        })
    except Exception as e:
        logger.error(f"Erro ao processar anotações: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dataset/upload-batch', methods=['POST'])
def api_upload_batch():
    """Permite fazer upload de várias imagens de uma vez para o dataset."""
    try:
        # Verificar se há arquivos anexados
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Nenhuma imagem enviada'
            }), 400
        
        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'Lista de arquivos vazia'
            }), 400
        
        # Criar diretórios se não existirem
        dataset_dir = 'dataset_treinamento'
        images_dir = os.path.join(dataset_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Contar uploads bem-sucedidos
        uploaded_count = 0
        errors = []
        
        # Processar cada arquivo
        for file in files:
            if file and file.filename:
                # Verificar se é uma imagem
                if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    errors.append(f"Arquivo {file.filename} não é uma imagem suportada")
                    continue
                
                try:
                    # Usar nome de arquivo seguro
                    filename = secure_filename(file.filename)
                    
                    # Se o arquivo já existir, adicionar timestamp ao nome
                    if os.path.exists(os.path.join(images_dir, filename)):
                        name, ext = os.path.splitext(filename)
                        timestamp = int(time.time())
                        filename = f"{name}_{timestamp}{ext}"
                    
                    # Salvar o arquivo
                    file_path = os.path.join(images_dir, filename)
                    file.save(file_path)
                    uploaded_count += 1
                    logger.info(f"Imagem salva para dataset: {file_path}")
                except Exception as e:
                    errors.append(f"Erro ao salvar {file.filename}: {str(e)}")
                    logger.error(f"Erro ao salvar imagem para dataset: {str(e)}")
        
        # Retornar resultado
        return jsonify({
            'success': True,
            'uploaded_count': uploaded_count,
            'total_files': len(files),
            'errors': errors if errors else None
        })
        
    except Exception as e:
        logger.error(f"Erro no upload em lote: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Inicializar o aplicativo
if __name__ == '__main__':
    # Tentar carregar um modelo padrão
    carregar_modelo()
    
    # Executar o servidor Flask
    porta = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=porta, debug=True) 