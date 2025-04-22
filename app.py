import cv2
import os
import json
import time
import uuid
import shutil
import sqlite3
import logging
import requests
import sys
import socket
import yaml
import base64
import traceback
import torch
import numpy as np
import warnings  # Adicionar importação do warnings
from datetime import datetime  # Mudar importação para usar datetime.now() diretamente
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from threading import Thread
from ultralytics import YOLO
from dotenv import load_dotenv  # Adicionar importação do load_dotenv
import io
from PIL import Image

# Sistema de Detecção de EPIs usando YOLOv8
# Baseado na biblioteca Ultralytics: https://github.com/ultralytics/ultralytics
# Versão: 1.0
# Autor: Sistema desenvolvido com ajuda do Claude

# Ignorar avisos desnecessários
warnings.filterwarnings('ignore')

# Carregar variáveis de ambiente
load_dotenv()

# Definir variáveis globais logo no início
app = Flask(__name__, static_folder='static')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'uma_chave_secreta_muito_segura')

# Configuração da pasta de uploads
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limitar uploads a 16 MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
print(f"✅ Pasta de uploads configurada em: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")

# Variável global para armazenar o detector
global detector
detector = None

# Definir YOLO_MODELS_BASE
YOLO_MODELS_BASE = {
    'yolov8n': 'yolov8n.pt',
    'yolov8s': 'yolov8s.pt',
    'yolov8m': 'yolov8m.pt',
    'yolov8l': 'yolov8l.pt',
    'yolov8x': 'yolov8x.pt'
}

# Inicializar YOLO_MODELS vazio - será preenchido com os modelos encontrados
YOLO_MODELS = {}

# VERIFICAÇÃO DE LOG: mostra todas as rotas disponíveis na inicialização
print("\n===== VERIFICANDO ROTAS DISPONÍVEIS =====")
print("O servidor foi reiniciado com o app.py atualizado contendo as rotas de processamento em lote:")
print("- /upload_multiple_images")
print("- /get_next_annotation_image")
print("- /save_annotation_and_next")
print("==========================================\n")

# Configurações
DATABASE = 'epi_detections.db'
NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL', 'admin@empresa.com')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')

# URLs dos modelos disponíveis (agora apenas para download inicial)
YOLO_MODELS = {
    # Modelos online
    'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
    'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
    'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
    'yolov8n-pose': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt',
    'yolo11n': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
    
    # Modelos locais (usar prefixo 'local:' para diferenciar de URLs)
    'local_default': 'local:model.pt',
    'epi_detector': 'local:runs/detect/epi_detector/weights/best.pt',
    'epi_detector2': 'local:runs/detect/epi_detector2/weights/best.pt'
}

# Variável global para modelos, será preenchida no __main__
YOLO_MODELS = {}

def download_model(model_name, target_path):
    """
    Baixa um modelo do repositório YOLO.
    Retorna True se o download for bem-sucedido, False caso contrário.
    """
    if model_name not in YOLO_MODELS_BASE:
        print(f"❌ Modelo {model_name} não encontrado no repositório de modelos.")
        return False
    
    url = YOLO_MODELS_BASE[model_name]
    print(f"Baixando modelo {model_name} de {url}...")
    
    try:
        # Criar diretório de destino se necessário
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ Modelo {model_name} baixado com sucesso para {target_path}")
        return True
    
    except Exception as e:
        print(f"❌ Erro ao baixar modelo {model_name}: {str(e)}")
        return False

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                results TEXT,
                all_epis_present BOOLEAN
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                message TEXT,
                sent BOOLEAN DEFAULT FALSE
            )
        ''')

def save_detection(image_path, results):
    """
    Salva uma detecção no banco de dados.
    
    Args:
        image_path: Caminho da imagem analisada
        results: Dicionário com resultados da análise
    """
    try:
        # Garantir que temos um dicionário de resultados válido
        if not isinstance(results, dict):
            print(f"❌ Erro ao salvar detecção: results não é um dicionário válido")
            return False
            
        # Converter o dicionário para JSON e salvar no banco de dados
        with sqlite3.connect(DATABASE) as conn:
            conn.execute(
                'INSERT INTO detections (image_path, results, all_epis_present) VALUES (?, ?, ?)',
                (image_path, json.dumps(results), results.get('all_epis_present', False))
            )
            
        print(f"✅ Detecção salva no banco de dados para imagem: {os.path.basename(image_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao salvar detecção no banco de dados: {str(e)}")
        traceback.print_exc()
        return False

def create_notification(message):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute(
            'INSERT INTO notifications (message, sent) VALUES (?, ?)',
            (message, False)
        )

def send_email_notification(subject, message):
    # Função desabilitada temporariamente
    print(f"[Email desabilitado] Assunto: {subject}, Mensagem: {message}")
    return True  # Simula sucesso

def process_notifications():
    while True:
        with sqlite3.connect(DATABASE) as conn:
            notifications = conn.execute(
                'SELECT id, message FROM notifications WHERE sent = 0'
            ).fetchall()
            
            for notification_id, message in notifications:
                if send_email_notification('Alerta de EPI', message):
                    conn.execute(
                        'UPDATE notifications SET sent = 1 WHERE id = ?',
                        (notification_id,)
                    )
        
        time.sleep(60)  # Verificar a cada minuto

# Inicializar banco de dados
init_db()

# Iniciar thread de notificações
notification_thread = Thread(target=process_notifications, daemon=True)
notification_thread.start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_models', methods=['GET'])
def get_available_models():
    print("\n--- DEBUG: Iniciando rota /get_models ---")
    global detector
    
    try:
        # Garantir que temos modelos disponíveis
        if not YOLO_MODELS:
            print("⚠️ YOLO_MODELS vazio, tentando redescobrir...")
            discovered = discover_local_models()
            YOLO_MODELS.update(discovered)
            if not YOLO_MODELS:
                print("⚠️ Nenhum modelo local encontrado, adicionando modelos base...")
                YOLO_MODELS.update(YOLO_MODELS_BASE)
        
        # Tentar inicializar detector se necessário
        if detector is None:
            print("⚠️ Detector não inicializado. Tentando inicializar...")
            initialize_detector()
        
        # Preparar resposta
        response = {
            'status': 'success',
            'current_model': None,
            'available_models': list(YOLO_MODELS.keys()),
            'classes': []
        }
        
        # Adicionar informações do detector se disponível
        if detector and hasattr(detector, 'current_model'):
            response['current_model'] = detector.current_model
            try:
                response['classes'] = detector.get_current_model_classes()
            except Exception as class_error:
                print(f"⚠️ Erro ao obter classes: {str(class_error)}")
                response['classes'] = []
        
        print(f"✅ Resposta /get_models preparada:")
        print(f"   - Modelo atual: {response['current_model']}")
        print(f"   - Modelos disponíveis: {len(response['available_models'])}")
        print(f"   - Classes: {len(response['classes'])}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Erro em get_models: {str(e)}")
        traceback.print_exc()
        
        # Garantir uma resposta válida mesmo em caso de erro
        return jsonify({
            'status': 'error',
            'message': str(e),
            'current_model': None,
            'available_models': list(YOLO_MODELS.keys()) if YOLO_MODELS else [],
            'classes': []
        })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analisa uma imagem e detecta objetos.
    """
    try:
        print("\n🔍 Iniciando análise de imagem")

        # 1. Verificar detector e modelo
        if not detector:
            print("❌ Detector não inicializado!")
            return jsonify({
                'status': 'error',
                'message': 'Sistema não inicializado. Tente recarregar a página.'
            }), 500

        if not detector.current_model:
            print("❌ Nenhum modelo ativo!")
            return jsonify({
                'status': 'error',
                'message': 'Nenhum modelo ativo. Selecione um modelo primeiro.'
            }), 500

        print(f"✅ Detector pronto. Modelo atual: {detector.current_model}")

        # 2. Verificar imagem
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Nenhuma imagem enviada'
            }), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Nome de arquivo vazio'
            }), 400

        # 3. Salvar imagem original
        filename = secure_filename(image_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        print(f"💾 Salvando imagem original em: {image_path}")
        image_file.save(image_path)

        if not os.path.exists(image_path):
            return jsonify({
                'status': 'error',
                'message': 'Falha ao salvar imagem'
            }), 500

        # 4. Verificar tamanho da imagem
        file_size = os.path.getsize(image_path)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            os.remove(image_path)
            return jsonify({
                'status': 'error',
                'message': f'Imagem muito grande: {file_size/1024/1024:.1f}MB (máximo 10MB)'
            }), 413

        # 5. Realizar detecção
        print(f"🔍 Executando detecção em: {image_path}")
        detections, img_with_boxes = detector.detect(image_path)

        if detections is None:
            return jsonify({
                'status': 'error',
                'message': 'Erro durante a detecção'
            }), 500
            
        # 6. Salvar imagem processada
        result_filename = None
        if img_with_boxes is not None:
            try:
                result_filename = f"result_{unique_filename}"
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                print(f"💾 Salvando imagem processada em: {result_path}")
                
                # Garantir que a imagem está no formato correto
                if isinstance(img_with_boxes, np.ndarray):
                    success = cv2.imwrite(result_path, img_with_boxes)
                    if not success:
                        print("❌ cv2.imwrite retornou False ao salvar a imagem")
                        result_filename = None
                    elif not os.path.exists(result_path):
                        print("❌ Arquivo não existe após cv2.imwrite")
                        result_filename = None
                    else:
                        print(f"✅ Imagem processada salva com sucesso: {os.path.getsize(result_path)} bytes")
                else:
                    print(f"❌ Formato inválido de img_with_boxes: {type(img_with_boxes)}")
                    result_filename = None
            except Exception as save_error:
                print(f"❌ Erro ao salvar imagem processada: {str(save_error)}")
                traceback.print_exc()
                result_filename = None
        else:
            print("⚠️ Nenhuma imagem processada para salvar (img_with_boxes é None)")

        # 7. Preparar resposta
        detected_classes = list(set(det['class_name'] for det in detections))
        print(f"📊 Classes detectadas: {detected_classes}")
        print(f"📊 Total de detecções: {len(detections)}")
            
        response = {
            'status': 'success',
            'message': 'Análise concluída com sucesso',
            'image_path': unique_filename,  # Enviar apenas o nome do arquivo
            'image_with_boxes_path': result_filename,  # Enviar apenas o nome do arquivo
            'detections_count': len(detections),
            'detections': detections,
            'detected_classes': detected_classes
        }

        # 8. Salvar no banco de dados
        try:
            db_results = {
                'detections_count': len(detections),
                'detected_classes': detected_classes,
                'detections': detections
            }
            save_detection(image_path, db_results)
        except Exception as db_error:
            print(f"⚠️ Erro ao salvar no banco de dados: {str(db_error)}")

        # Normalizar classes selecionadas se existirem
        if 'selected_classes' in request.form:
            selected_classes = json.loads(request.form['selected_classes'])
            selected_classes = [detector._normalize_class_name(cls) for cls in selected_classes]
            print(f"📋 Classes selecionadas (normalizadas): {selected_classes}")

        return jsonify(response)

    except Exception as e:
        print(f"❌ Erro na análise: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Erro interno: {str(e)}'
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    print("\n--- DEBUG: Iniciando rota /stats ---")
    try:
        # Inicializa dicionário de resultados com valores padrão
        results = {
            'total_detections': 0,
            'detections_by_class': {},
            'total_images': 0,
            'images_with_detections': 0,
            'images_without_detections': 0,
            'compliance_rate': 0.0  # Garantir que é float
        }
        
        # Conecta ao banco de dados
        conn = sqlite3.connect('epi_detections.db')
        cursor = conn.cursor()
        
        try:
            # Total de imagens analisadas
            cursor.execute('SELECT COUNT(*) FROM detections')
            results['total_images'] = int(cursor.fetchone()[0])
            
            # Imagens com e sem detecções
            cursor.execute('''
                SELECT COUNT(*) as count,
                CASE 
                    WHEN results IS NOT NULL AND json_valid(results) AND json_array_length(json_extract(results, '$.raw_detections_count')) > 0 THEN 1
                    ELSE 0 
                END as has_detections
                FROM detections
                GROUP BY has_detections
            ''')
            
            for count, has_detections in cursor.fetchall():
                if has_detections:
                    results['images_with_detections'] = int(count)
                else:
                    results['images_without_detections'] = int(count)
            
            # Contagem de detecções por classe
            cursor.execute('SELECT results FROM detections WHERE results IS NOT NULL AND json_valid(results)')
            total_detections = 0
            
            for (results_json,) in cursor.fetchall():
                try:
                    detection_data = json.loads(results_json)
                    if isinstance(detection_data, dict):
                        # Novo formato (experimento)
                        if 'experiment' in detection_data and detection_data.get('found') is not None:
                            target_class = detection_data.get('target_class', 'pessoa')
                            if detection_data['found']:
                                results['detections_by_class'][target_class] = int(results['detections_by_class'].get(target_class, 0) + 1)
                                total_detections += 1
                        
                        # Formato antigo
                        elif 'raw_detections_count' in detection_data:
                            count = int(detection_data['raw_detections_count'])
                            total_detections += count
                            results['detections_by_class']['total'] = int(results['detections_by_class'].get('total', 0) + count)
                
                except json.JSONDecodeError as e:
                    print(f"⚠️ Erro ao decodificar JSON de detecções: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ Erro ao processar detecção: {e}")
                    continue
            
            results['total_detections'] = int(total_detections)
            
            # Calcular taxa de conformidade
            if results['total_images'] > 0:
                results['compliance_rate'] = round(float(results['images_with_detections']) / float(results['total_images']) * 100.0, 2)
            else:
                results['compliance_rate'] = 0.0
            
            print(f"📊 Estatísticas calculadas:")
            print(f"   - Total de imagens: {results['total_images']}")
            print(f"   - Com detecções: {results['images_with_detections']}")
            print(f"   - Sem detecções: {results['images_without_detections']}")
            print(f"   - Taxa de conformidade: {results['compliance_rate']}%")
            
        except sqlite3.Error as sql_error:
            print(f"⚠️ Erro SQL em /stats: {sql_error}")
            # Não propagar o erro, manter os resultados padrão
        
        finally:
            conn.close()
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        print(f"❌ Erro em /stats: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'results': {
                'total_detections': 0,
                'detections_by_class': {},
                'total_images': 0,
                'images_with_detections': 0,
                'images_without_detections': 0,
                'compliance_rate': 0.0  # Garantir que é float mesmo em caso de erro
            }
        }), 500

@app.route('/diagnostico', methods=['GET'])
def diagnostico_sistema():
    """Retorna informações de diagnóstico sobre o estado do sistema."""
    status_info = {
        "status": "success",
        "message": "Informações de diagnóstico",
        "detector_inicializado": detector is not None,
        "modelo_atual": detector.current_model if detector else "N/A",
        "classes_modelo_atual": detector.get_current_model_classes() if detector else [],
        "modelos_disponiveis": list(YOLO_MODELS.keys()),
        "diretorio_upload": app.config.get('UPLOAD_FOLDER', 'N/A'),
        "database_path": os.path.abspath(DATABASE),
        "dataset_path_local": os.path.abspath('dataset_treinamento'),
        "python_version": sys.version,
        "torch_version": torch.__version__ if torch else "N/A",
        "cv2_version": cv2.__version__ if cv2 else "N/A",
        "ultralytics_version": "Verificar manualmente" # Evitar importação aqui
    }
    return jsonify(status_info)

@app.route('/manual_override', methods=['POST'])
def manual_override():
    """Rota para definir manualmente os EPIs presentes na imagem."""
    try:
        data = request.json
        image_path = data.get('image_path')
        epi_states = data.get('epi_states', {})
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Imagem não encontrada'}), 404
        
        # Atualizar o banco de dados com a avaliação manual
        with sqlite3.connect(DATABASE) as conn:
            # Obter o registro atual
            record = conn.execute(
                'SELECT id, results FROM detections WHERE image_path = ?',
                (image_path,)
            ).fetchone()
            
            if not record:
                return jsonify({'error': 'Registro não encontrado'}), 404
            
            id, results_json = record
            results = json.loads(results_json)
            
            # Atualizar os resultados com a avaliação manual
            for epi, present in epi_states.items():
                if epi in results['results']:
                    results['results'][epi] = present
            
            # Recalcular all_epis_present
            results['all_epis_present'] = all(results['results'].values())
            
            # Atualizar o registro
            conn.execute(
                'UPDATE detections SET results = ?, all_epis_present = ? WHERE id = ?',
                (json.dumps(results), results['all_epis_present'], id)
            )
        
        return jsonify({
            'status': 'success',
            'message': 'Registro atualizado com sucesso',
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/set_model/<model_name>', methods=['GET'])
def set_model(model_name):
    """Permite trocar o modelo em tempo de execução e retorna as classes do novo modelo."""
    try:
        print(f"\n🔄 Solicitação para trocar modelo para: {model_name}")
        
        # Verificar se o modelo existe
        if model_name not in YOLO_MODELS:
            print(f"❌ Modelo {model_name} não reconhecido")
            return jsonify({
                'status': 'error',
                'message': f'Modelo {model_name} não reconhecido. Opções: {list(YOLO_MODELS.keys())}'
            }), 400

        # Resetar o detector para garantir uma troca limpa
        global detector
        detector = None
        detector = YoloEPIDetector()
        
        # Tentar carregar o novo modelo
        print(f"🔄 Tentando carregar modelo: {model_name}")
        if not detector.load_model(model_name):
            print(f"❌ Falha ao carregar modelo {model_name}")
            return jsonify({
                'status': 'error',
                'message': f'Erro ao carregar modelo {model_name}. Veja os logs do servidor.'
            }), 500

        # Configurar classes do modelo
        detector._configure_class_mapping()
        
        # Obter classes do novo modelo
        classes = detector.get_current_model_classes()
        print(f"✅ Modelo {model_name} carregado com sucesso")
        print(f"📋 Classes disponíveis: {classes}")

        return jsonify({
            'status': 'success',
            'message': f'Modelo alterado para {model_name}',
            'model': model_name,
            'classes': classes
        })
            
    except Exception as e:
        print(f"❌ Erro ao processar /set_model/{model_name}: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Erro: {str(e)}'
        }), 500

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve imagens do diretório de uploads."""
    try:
        # Criar o diretório de uploads se não existir
        if not os.path.exists('uploads'):
            os.makedirs('uploads', exist_ok=True)
            print(f"⚠️ Diretório de uploads criado")
        
        # Caminho completo para o arquivo
        filepath = os.path.join('uploads', filename)
        
        # Verificar se o arquivo existe
        if not os.path.exists(filepath):
            print(f"❌ Imagem não encontrada: {filepath}")
            return "Imagem não encontrada", 404
        
        # Servir o arquivo como resposta
        return send_file(filepath)
    except Exception as e:
        print(f"❌ Erro ao servir imagem {filename}: {str(e)}")
        return f"Erro ao servir imagem: {str(e)}", 500

@app.route('/treinar', methods=['GET'])
def pagina_treinamento():
    """Rota para a página de treinamento do modelo de detecção de EPIs."""
    try:
        # Verificar se a pasta de treinamento existe e criar a estrutura completa
        os.makedirs('dataset_treinamento', exist_ok=True)
        os.makedirs('dataset_treinamento/images', exist_ok=True)
        os.makedirs('dataset_treinamento/labels', exist_ok=True)
        
        # Verificar o caminho absoluto do dataset (para diagnóstico)
        dataset_path = os.path.abspath('dataset_treinamento')
        print(f"✅ Dataset path: {dataset_path}")
        print(f"   - Images: {os.path.join(dataset_path, 'images')}")
        print(f"   - Labels: {os.path.join(dataset_path, 'labels')}")
        
        # Contar quantas imagens já existem
        num_imagens = len(os.listdir('dataset_treinamento/images'))
        print(f"   - Total de imagens: {num_imagens}")
        
        # Verificar permissões de diretório
        try:
            # Tentar criar um arquivo de teste para verificar permissões
            test_file = os.path.join('dataset_treinamento', 'test_write.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print("✅ Permissões de escrita OK")
        except Exception as e:
            print(f"⚠️ Aviso: Possível problema de permissões no diretório: {str(e)}")
        
        return render_template('treinamento.html', num_imagens=num_imagens)
    except Exception as e:
        print(f"❌ Erro ao acessar página de treinamento: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro ao acessar página de treinamento: {str(e)}'
        }), 500

@app.route('/salvar_anotacao', methods=['POST'])
def salvar_anotacao():
    """Salva uma imagem e suas anotações para o dataset de treinamento."""
    try:
        # Verificar se recebemos a imagem
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400
        
        # Garantir que os diretórios existam
        os.makedirs('dataset_treinamento/images', exist_ok=True)
        os.makedirs('dataset_treinamento/labels', exist_ok=True)
        
        # Obter a imagem
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Converter para RGB se necessário
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            image = image.convert('RGB')
        
        # Obter as anotações (bounding boxes) do formulário
        anotacoes = request.form.get('annotations', '[]')
        try:
            anotacoes = json.loads(anotacoes)
        except:
            return jsonify({'error': 'Formato inválido de anotações'}), 400
        
        # Verificar se temos anotações válidas
        if not isinstance(anotacoes, list) or len(anotacoes) == 0:
            return jsonify({'error': 'Nenhuma anotação encontrada'}), 400
        
        # Criar nome de arquivo com timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_filename = f"{timestamp}.jpg"
        txt_filename = f"{timestamp}.txt"
        
        # Caminhos completos para os arquivos
        image_path = os.path.join('dataset_treinamento', 'images', img_filename)
        label_path = os.path.join('dataset_treinamento', 'labels', txt_filename)
        
        # Salvar a imagem
        image.save(image_path)
        print(f"✅ Imagem salva em: {image_path}")
        
        # Salvar as anotações no formato YOLO
        # Formato: <class_id> <x_center> <y_center> <width> <height>
        # Valores normalizados entre 0 e 1
        with open(label_path, 'w') as f:
            for anotacao in anotacoes:
                classe_id = anotacao.get('class_id', 0)
                x = anotacao.get('x', 0)
                y = anotacao.get('y', 0)
                w = anotacao.get('width', 0)
                h = anotacao.get('height', 0)
                
                # Escrever no formato YOLO
                f.write(f"{classe_id} {x} {y} {w} {h}\n")
        
        print(f"✅ Anotações salvas em: {label_path}")
        
        # Verificar quantas imagens já temos no dataset
        num_imagens = len(os.listdir('dataset_treinamento/images'))
        
        return jsonify({
            'status': 'success',
            'message': 'Imagem e anotações salvas com sucesso',
            'filename': img_filename,
            'num_imagens': num_imagens
        })
    except Exception as e:
        print(f"❌ Erro ao salvar anotação: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Erro ao salvar anotação: {str(e)}'
        }), 500

@app.route('/iniciar_treinamento', methods=['POST'])
def iniciar_treinamento():
    """Inicia o treinamento do modelo YOLOv8 com o dataset atual."""
    try:
        # Obter parâmetros da requisição
        data = request.json
        epochs = data.get('epochs', 100)
        batch_size = data.get('batch_size', 16)
        save_checkpoints = data.get('save_checkpoints', True)
        
        # Verificar se temos imagens suficientes
        num_imagens = len(os.listdir('dataset_treinamento/images'))
        if num_imagens < 10:
            return jsonify({
                'status': 'error',
                'message': f'Você precisa de pelo menos 10 imagens para treinar. Você tem apenas {num_imagens}.'
            }), 400
        
        # Verificar configuração do YOLO
        datasets_dir = os.path.join(os.path.expanduser('~'), 'Visao', 'datasets')
        target_dataset_dir = os.path.join(datasets_dir, 'dataset_treinamento')
        
        # Sincronizar os diretórios
        setup_dataset_directories()
        
        # Criar arquivo de configuração para treinamento
        config = {
            'path': target_dataset_dir,  # Usar o diretório que o YOLO espera
            'train': './images',  # Caminho relativo ao dataset_path
            'val': './images',    # Caminho relativo ao dataset_path
            'names': {
                0: 'pessoa',
                1: 'capacete',
                2: 'oculos_protecao',
                3: 'mascara',
                4: 'luvas',
                5: 'colete'
            }
        }
        
        # Verificar se o diretório de destino existe
        if not os.path.exists(target_dataset_dir):
            return jsonify({
                'status': 'error',
                'message': f'Erro: Diretório de dataset não configurado corretamente: {target_dataset_dir}'
            }), 500
        
        # Verificar se temos imagens no diretório de destino
        target_images_dir = os.path.join(target_dataset_dir, 'images')
        if not os.path.exists(target_images_dir) or len(os.listdir(target_images_dir)) < 1:
            return jsonify({
                'status': 'error',
                'message': f'Erro: Não há imagens no diretório de destino. Tente adicionar imagens novamente.'
            }), 500
        
        # Salvar a configuração em um arquivo
        config_path = os.path.abspath('dataset_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"✅ Arquivo de configuração criado em: {config_path}")
        print(f"   - Dataset path: {target_dataset_dir}")
        print(f"   - Número de imagens: {num_imagens}")
        print(f"   - Parâmetros de treinamento: {epochs} épocas, batch size {batch_size}")
        
        # Iniciar treinamento em thread separada para não bloquear o servidor
        import threading
        
        def treinar_modelo():
            try:
                print("🔄 Iniciando treinamento do modelo...")
                from ultralytics import YOLO
                
                # Carregar modelo base
                model = YOLO('yolov8n.pt')
                
                # Iniciar treinamento
                results = model.train(
                    data=config_path,
                    epochs=epochs,
                    imgsz=640,
                    batch=batch_size,
                    name='epi_detector',
                    patience=20,
                    save=save_checkpoints
                )
                
                print("✅ Treinamento concluído com sucesso!")
                print(f"   Modelo salvo em: {os.path.abspath('runs/detect/epi_detector')}")
                
                # Copiar o melhor modelo para ambos os locais
                import shutil
                best_model_path = os.path.join('runs', 'detect', 'epi_detector', 'weights', 'best.pt')
                if os.path.exists(best_model_path):
                    # Copiar para a raiz (compatibilidade com código anterior)
                    shutil.copy(best_model_path, 'epi_detector.pt')
                    print("✅ Modelo copiado para epi_detector.pt na raiz do projeto")
                    
                    # Garantir que o diretório base também tenha uma cópia atualizada
                    # (não precisamos fazer nada, pois já está salvo corretamente no caminho original)
                    print("✅ Modelo pronto para uso em ambos os caminhos")
                    print("   1. runs/detect/epi_detector/weights/best.pt (original)")
                    print("   2. epi_detector.pt (cópia na raiz)")
                    print("   Para usar o modelo treinado, reinicie o servidor ou selecione-o na interface")
                else:
                    print("❌ Modelo treinado não encontrado no caminho esperado")
            except Exception as e:
                print(f"❌ Erro durante treinamento: {str(e)}")
        
        # Iniciar thread de treinamento
        thread = threading.Thread(target=treinar_modelo)
        thread.daemon = True
        thread.start()
        
        # Gerar ID de treinamento para acompanhamento
        training_id = f"{time.strftime('%Y%m%d%H%M%S')}"
        
        return jsonify({
            'success': True,
            'status': 'success',
            'message': 'Treinamento iniciado em segundo plano. Este processo pode demorar várias horas.',
            'num_imagens': num_imagens,
            'training_id': training_id
        })
    except Exception as e:
        print(f"❌ Erro ao iniciar treinamento: {str(e)}")
        return jsonify({
            'status': 'error',
            'success': False,
            'message': f'Erro ao iniciar treinamento: {str(e)}'
        }), 500

@app.route('/api/dataset/status', methods=['GET'])
def api_dataset_status():
    """Retorna informações sobre o status atual do dataset de treinamento."""
    try:
        # Verificar diretórios e criar se não existirem
        os.makedirs('dataset_treinamento/images', exist_ok=True)
        os.makedirs('dataset_treinamento/labels', exist_ok=True)
        
        # Contar imagens e anotações
        images_dir = os.path.join('dataset_treinamento', 'images')
        labels_dir = os.path.join('dataset_treinamento', 'labels')
        
        total_images = len(os.listdir(images_dir))
        total_labels = len(os.listdir(labels_dir))
        
        # Obter contagem por classe
        class_counts = {}
        for label_file in os.listdir(labels_dir):
            label_path = os.path.join(labels_dir, label_file)
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_name = get_class_name(class_id)
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            except Exception as e:
                print(f"Erro ao ler arquivo de anotação {label_file}: {str(e)}")
        
        return jsonify({
            'success': True,
            'total_images': total_images,
            'annotated_images': total_labels,
            'class_counts': class_counts,
            'ready_for_training': total_labels >= 10
        })
    except Exception as e:
        print(f"❌ Erro ao obter status do dataset: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dataset/images', methods=['GET'])
def api_dataset_images():
    """Retorna a lista de imagens no dataset."""
    try:
        images_dir = os.path.join('dataset_treinamento', 'images')
        labels_dir = os.path.join('dataset_treinamento', 'labels')
        
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
        
        images = []
        for img_file in os.listdir(images_dir):
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
                    print(f"Erro ao criar thumbnail para {img_file}: {str(e)}")
                
                images.append({
                    'id': len(images),
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
        print(f"❌ Erro ao obter imagens do dataset: {str(e)}")
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
        
        # Se não encontrar classes no arquivo, usar padrão
        if not classes:
            classes = ['pessoa', 'capacete', 'oculos_protecao', 'mascara', 'luvas', 'colete']
        
        return jsonify({
            'success': True,
            'classes': classes
        })
    except Exception as e:
        print(f"❌ Erro ao obter classes disponíveis: {str(e)}")
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
                print(f"❌ Erro ao salvar imagem: {str(e)}")
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
                        print(f"⚠️ Classe '{class_name}' não encontrada no config, usando 0")
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
            print(f"❌ Erro ao salvar anotações: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Erro ao salvar anotações: {str(e)}'
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Anotações salvas com sucesso'
        })
    except Exception as e:
        print(f"❌ Erro ao processar anotações: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dataset/upload-batch', methods=['POST'])
def api_upload_batch():
    """Faz upload de várias imagens em lote para o dataset."""
    try:
        # Verificar se existem arquivos na solicitação
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Nenhum arquivo enviado'
            }), 400
        
        # Obter arquivos
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return jsonify({
                'success': False,
                'error': 'Lista de arquivos vazia'
            }), 400
        
        # Criar diretórios se não existirem
        images_dir = os.path.join('dataset_treinamento', 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Contar uploads bem-sucedidos
        uploaded_count = 0
        
        # Processar cada arquivo
        for file in files:
            if file and file.filename:
                try:
                    # Gerar nome de arquivo seguro
                    filename = secure_filename(file.filename)
                    
                    # Verificar extensão
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        print(f"⚠️ Tipo de arquivo não suportado: {filename}")
                        continue
                    
                    # Salvar arquivo
                    file_path = os.path.join(images_dir, filename)
                    file.save(file_path)
                    
                    uploaded_count += 1
                except Exception as e:
                    print(f"❌ Erro ao salvar arquivo {file.filename}: {str(e)}")
        
        # Verificar se pelo menos um arquivo foi salvo
        if uploaded_count == 0:
            return jsonify({
                'success': False,
                'error': 'Nenhum arquivo pôde ser salvo'
            }), 400
        
        return jsonify({
            'success': True,
            'uploaded_count': uploaded_count,
            'message': f'{uploaded_count} imagens adicionadas ao dataset'
        })
        
    except Exception as e:
        print(f"❌ Erro no upload em lote: {str(e)}")
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
                        class_name = get_class_name(class_id)
                        
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
        print(f"❌ Erro ao obter imagem do dataset: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_class_name(class_id):
    """Retorna o nome da classe a partir do ID."""
    classes = {
        0: 'pessoa',
        1: 'capacete',
        2: 'oculos',
        3: 'protetor_auricular',
        4: 'mascara',
        5: 'luva',
        6: 'bota',
        7: 'colete'
    }
    return classes.get(class_id, f'classe_{class_id}')

@app.route('/detect_specific_epi', methods=['POST'])
def detect_specific_epi():
    """
    Rota para detecção específica de um tipo de EPI.
    Recebe o caminho da imagem e o tipo de EPI para detectar.
    
    Espera um JSON com:
    {
        "image_path": "caminho/para/imagem.jpg",
        "epi_type": "oculos_protecao"
    }
    
    Retorna:
    {
        "success": true/false,
        "detected": true/false,
        "confidence": float,
        "coordinates": [x1, y1, x2, y2] ou null,
        "message": "mensagem"
    }
    """
    data = request.json
    
    # Verificar se o JSON contém os campos necessários
    if not data or 'image_path' not in data or 'epi_type' not in data:
        return jsonify({
            'success': False,
            'message': 'Dados incompletos. Esperado: image_path e epi_type'
        }), 400

    image_path = data['image_path']
    epi_type = data['epi_type']
        
    # Verificar se o arquivo existe
    if not os.path.exists(image_path):
        return jsonify({
            'success': False,
            'message': f'Imagem não encontrada: {image_path}'
        }), 404
    
    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({
            'success': False,
            'message': f'Falha ao carregar a imagem: {image_path}'
        }), 500
            
    # Verificar o tipo de EPI solicitado
    if epi_type == 'oculos_protecao':
        # Usar o detector específico para óculos
        detector = get_detector()
        result = detector.detect_specific_glasses(image)
            
        return jsonify({
            'success': True,
            'detected': result['detected'],
            'confidence': float(result['confidence']),
            'coordinates': result['coordinates'],
            'message': 'Análise concluída com sucesso'
        })
    else:
        return jsonify({
            'success': False,
            'message': f'Tipo de EPI não suportado: {epi_type}. Atualmente suportamos apenas: oculos_protecao'
        }), 400

class YoloEPIDetector:
    """Detector de EPIs usando YOLOv8."""
    
    def __init__(self):
        # CACHE MODIFICADO: Agora armazena {nome_modelo: {path: caminho_absoluto}}
        self.models_cache = {}
        self.current_model = None # Apenas o NOME do modelo ativo
        # predictions_cache removido por enquanto
        self.coco_to_epi = {
            0: "pessoa", # Mapeamento inicial de fallback
        }
        self._initialize_cache_from_yolo_models()

    def _initialize_cache_from_yolo_models(self):
        """Preenche o cache inicial com os caminhos dos modelos de YOLO_MODELS."""
        print("\n--- Inicializando Cache de Modelos (paths) --- ")
        for name, source in YOLO_MODELS.items():
             if source.startswith("local:"):
                 path = os.path.abspath(source.replace("local:", ""))
                 if os.path.exists(path):
                      self.models_cache[name] = {"path": path}
                      print(f"   - Cache Add (Local): '{name}' -> '{path}'")
                 else:
                      print(f"   - Cache Warn (Local): Arquivo '{name}' não encontrado em '{path}'")
             elif source.startswith("http") or source.endswith(".pt"):
                  # Para modelos online ou nomes .pt, o caminho será determinado no download/load
                  # Apenas registrar que o modelo existe
                  self.models_cache[name] = {"path": None} # Path será definido após download
                  print(f"   - Cache Add (Online/Nome): '{name}' -> Path indefinido (será baixado)")
        print("--- Cache de Modelos (paths) Inicializado --- ")

    def load_model(self, model_to_load=None):
        """
        Carrega um modelo YOLO para detecção de EPIs.
        
        Args:
            model_to_load (str, opcional): Nome do modelo específico para carregar.
                Se None, tenta carregar o primeiro modelo disponível.
        
        Returns:
            bool: True se o modelo foi carregado com sucesso, False caso contrário.
        """
        try:
            print(f"\n🔄 Iniciando carregamento de modelo: {model_to_load}")
            
            # Se nenhum modelo específico foi solicitado, usar o primeiro disponível
            if not model_to_load:
                if not YOLO_MODELS:
                    print("❌ Nenhum modelo YOLO disponível para carregar")
                    return False
                model_to_load = next(iter(YOLO_MODELS))
            
            # Verificar se o modelo solicitado existe
            if model_to_load not in YOLO_MODELS:
                print(f"❌ Modelo '{model_to_load}' não encontrado em YOLO_MODELS")
                return False
                
            model_path = YOLO_MODELS[model_to_load]
            print(f"🔄 Tentando carregar modelo: {model_to_load} de {model_path}")
            
            # Se for um modelo local, verificar se o arquivo existe
            if isinstance(model_path, str) and model_path.startswith('local:'):
                local_path = model_path.replace('local:', '')
                if not os.path.exists(local_path):
                    print(f"❌ Arquivo do modelo não encontrado: {local_path}")
                    return False
                model_path = local_path
            
            # Limpar modelo anterior se existir
            if hasattr(self, 'model'):
                del self.model
                
            # Carregar o novo modelo
            print(f"🔄 Carregando modelo de: {model_path}")
            self.model = YOLO(model_path)
            self.current_model = model_to_load
            
            # Configurar mapeamento de classes
            print("🔄 Configurando mapeamento de classes...")
            self._configure_class_mapping()
            
            print(f"✅ Modelo '{model_to_load}' carregado com sucesso")
            print(f"📋 Classes disponíveis: {self.get_current_model_classes()}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo '{model_to_load}': {str(e)}")
            traceback.print_exc()
            # Limpar estado em caso de erro
            self.model = None
            self.current_model = None
            return False

    def _update_class_mapping(self, model_object=None):
        """
        Atualiza o mapeamento de classes (self.coco_to_epi) com base no self.current_model.
        Tenta ler de args.yaml se o modelo for local.
        Aplica mapeamento específico para yolo11.
        Fallback para COCO se nada for encontrado.

        Args:
            model_object: Ignorado nesta versão (passado como None).
        """
        try:
            current_model_name = self.current_model # USA O NOME ATUAL
            print(f"\n--- Iniciando _update_class_mapping para modelo ATUAL: {current_model_name} (sem objeto model) ---")

            if not current_model_name:
                 print("   - ⚠️ Nenhum modelo atual definido. Usando fallback mínimo.")
                 self.coco_to_epi = {0: "desconhecido"}
                 print(f"--- FINALIZANDO _update_class_mapping (sem modelo atual). Mapeamento: {self.coco_to_epi} ---")
                 return

            # Obter informações do cache
            model_info = self.models_cache.get(current_model_name, {})
            model_path = model_info.get("path") # Pode ser None para modelos não baixados ainda

            # --- DEBUGGING YOLO11 CHECK ---
            is_yolo11 = False
            current_model_name_lower = ""
            if current_model_name:
                current_model_name_lower = current_model_name.lower()
                is_yolo11 = 'yolo11' in current_model_name_lower
            print(f"    [DEBUG] Nome do modelo: '{current_model_name}' (tipo: {type(current_model_name)})")
            print(f"    [DEBUG] Nome em minúsculas: '{current_model_name_lower}' (tipo: {type(current_model_name_lower)})")
            print(f"    [DEBUG] Verificação ('yolo11' in nome_minusculo): {is_yolo11}")
            # --- END DEBUGGING ---

            # 1. Verificar se é o modelo YOLO11 (PRIORIDADE)
            if is_yolo11:
                print(f"✅ Detectado modelo {current_model_name}, aplicando mapeamento específico de classes EPI.")
                self.coco_to_epi = {
                    0: "pessoa", 1: "capacete", 2: "colete", 3: "luvas",
                    4: "mascara", 5: "oculos", 6: "protetor_auricular"
                }
                print(f"🏷️ Mapeamento DEFINIDO para {current_model_name}: {self.coco_to_epi}")
                print(f"--- FINALIZANDO _update_class_mapping (sucesso via verificação de nome {current_model_name}) ---")
                return # Sair imediatamente

            # 2. Se NÃO for YOLO11, tentar ler do args.yaml (se houver caminho)
            print(f"ℹ️ Modelo {current_model_name} não é YOLO11. Tentando obter classes de args.yaml...")
            model_classes_from_yaml = {}
            if model_path and os.path.exists(model_path):
                model_dir = os.path.dirname(model_path)
                args_path = os.path.join(model_dir, 'args.yaml')
                print(f"   - Procurando args.yaml em: {args_path}")
                if os.path.exists(args_path):
                    try:
                        with open(args_path, 'r') as f:
                            args_data = yaml.safe_load(f)
                        if args_data and 'names' in args_data and args_data['names']:
                            names_data = args_data['names']
                            if isinstance(names_data, dict):
                                try:
                                    model_classes_from_yaml = {int(k): v for k, v in names_data.items()}
                                    print(f"✅ Classes extraídas de args.yaml (dict): {model_classes_from_yaml}")
                                except ValueError:
                                    print(f"   - Erro: Chaves do dict em args.yaml não são inteiros: {names_data.keys()}")
                            elif isinstance(names_data, list):
                                model_classes_from_yaml = {i: name for i, name in enumerate(names_data)}
                                print(f"✅ Classes extraídas de args.yaml (list): {model_classes_from_yaml}")
                            else:
                                print(f"   - Tipo inesperado para 'names' em args.yaml: {type(names_data)}")
                    except Exception as e:
                        print(f"⚠️ Erro ao ler/processar args.yaml: {str(e)}")
                        traceback.print_exc()
                else:
                    print(f"   - Arquivo args.yaml não encontrado.")
            else:
                 print(f"   - Caminho do modelo ('{model_path}') inválido ou não definido no cache. Não é possível procurar args.yaml.")

            # 3. Definir o mapeamento
            if model_classes_from_yaml:
                self.coco_to_epi = model_classes_from_yaml
                print(f"🏷️ Mapeamento definido via args.yaml: {self.coco_to_epi}")
            else:
                # 4. Fallback para COCO (se não for YOLO11 e não achou em args.yaml)
                print("⚠️ Nenhuma classe encontrada em args.yaml. Usando mapeamento padrão COCO.")
                self.coco_to_epi = { # COCO classes
                     0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
                     5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
                     10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
                     14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
                     20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
                     25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
                     30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
                     35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
                     39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
                     45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
                     50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
                     56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
                     61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
                     67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
                     72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
                     77: "teddy bear", 78: "hair drier", 79: "toothbrush"
                }
                print(f"🏷️ Mapeamento definido via fallback COCO: {self.coco_to_epi}")

            # Garantir que é dicionário com chaves int
            if not isinstance(self.coco_to_epi, dict):
                 print(f"❌ Mapeamento inválido ou vazio ({type(self.coco_to_epi)}). Resetando para fallback mínimo.")
                 self.coco_to_epi = {0: "desconhecido"}
            elif any(not isinstance(k, int) for k in self.coco_to_epi.keys()):
                 print(f"⚠️ Chaves do mapeamento não são inteiros ({list(self.coco_to_epi.keys())}). Corrigindo...")
                 try:
                     self.coco_to_epi = {int(k): v for k, v in self.coco_to_epi.items()}
                 except Exception as e:
                      print(f"❌ Erro ao converter chaves para int: {str(e)}. Resetando para fallback mínimo.")
                      self.coco_to_epi = {0: "desconhecido"}

            print(f"🏷️ Mapeamento final de classes (_update_class_mapping para {current_model_name}): {self.coco_to_epi}")
            print(f"--- FINALIZANDO _update_class_mapping (lógica padrão para {current_model_name}) ---")

        except Exception as e:
            print(f"❌ Erro GERAL em _update_class_mapping: {str(e)}")
            traceback.print_exc()
            print("⚠️ Resetando mapeamento para fallback mínimo devido a erro.")
            self.coco_to_epi = {0: "desconhecido"}

    def detect(self, image_path):
        """
        Realiza a detecção de objetos em uma imagem.
        """
        try:
            print(f"\n🔍 Iniciando detecção em: {image_path}")
            
            # 1. Verificar se temos um modelo atual
            if not self.current_model:
                print("❌ Erro: Nenhum modelo atual definido")
                return [], None
                
            # 2. Obter caminho do modelo do cache
            model_info = self.models_cache.get(self.current_model)
            if not model_info or not model_info.get("path"):
                print(f"❌ Erro: Caminho do modelo '{self.current_model}' não encontrado no cache")
                return [], None
                
            model_path = model_info["path"]
            print(f"✅ Usando modelo: {model_path}")
            
            # 3. Verificar se a imagem existe
            if not os.path.exists(image_path):
                print(f"❌ Erro: Imagem não encontrada: {image_path}")
                return [], None
                
            # 4. Carregar o modelo YOLO
            try:
                print(f"🔄 Carregando modelo YOLO de: {model_path}")
                model = YOLO(model_path)
                print("✅ Modelo YOLO carregado com sucesso")
            except Exception as e:
                print(f"❌ Erro ao carregar modelo YOLO: {str(e)}")
                traceback.print_exc()
                return [], None
            
            # 5. Executar a detecção com parâmetros ajustados
            print("🔍 Executando detecção...")
            results = model(image_path, conf=0.15, iou=0.45)  # Reduzir confiança e IOU para detectar mais objetos
            
            if not results:
                print("⚠️ Nenhum resultado retornado pelo modelo")
                return [], None
                
            # 6. Processar resultados
            detections = []
            result = results[0]  # Pegar primeiro resultado
            
            # Verificar se temos boxes
            if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                print("⚠️ Nenhuma detecção encontrada na imagem")
                return [], None
            
            # 7. Gerar imagem com caixas
            try:
                print("🎨 Gerando imagem com caixas delimitadoras...")
                img_with_boxes = result.plot()
                
                # Verificar se a imagem foi gerada corretamente
                if img_with_boxes is None:
                    print("❌ Falha ao gerar imagem com caixas")
                    return [], None
                
                # Converter BGR para RGB se necessário
                if len(img_with_boxes.shape) == 3 and img_with_boxes.shape[2] == 3:
                    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
                
                print(f"✅ Imagem com caixas gerada. Shape: {img_with_boxes.shape}")
            except Exception as box_error:
                print(f"❌ Erro ao gerar imagem com caixas: {str(box_error)}")
                traceback.print_exc()
                return [], None
            
            # 8. Processar cada detecção
            for i, box in enumerate(result.boxes):
                try:
                    # Extrair dados da caixa
                    class_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Obter nome da classe e normalizar
                    class_name = self.coco_to_epi.get(class_id, f"classe_{class_id}")
                    normalized_class_name = self._normalize_class_name(class_name)
                    
                    detection = {
                        "id": i,
                        "class_id": class_id,
                        "class_name": normalized_class_name,
                        "original_class_name": class_name,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2]
                    }
                    detections.append(detection)
                    print(f"  ✅ Detecção {i}: {normalized_class_name} ({conf:.3f}) em [{x1}, {y1}, {x2}, {y2}]")
                    
                except Exception as box_error:
                    print(f"⚠️ Erro ao processar caixa {i}: {str(box_error)}")
                    continue
            
            print(f"✅ Processadas {len(detections)} detecções com sucesso")
            
            # 9. Adicionar informações na imagem
            if img_with_boxes is not None and len(detections) > 0:
                try:
                    # Adicionar texto com informações
                    height = img_with_boxes.shape[0]
                    text = f"Modelo: {self.current_model} | Detecções: {len(detections)}"
                    cv2.putText(img_with_boxes, 
                              text,
                              (10, height - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, 
                              (255, 255, 255), 
                              2)
                    print("✅ Texto adicionado à imagem")
                except Exception as text_error:
                    print(f"⚠️ Erro ao adicionar texto na imagem: {str(text_error)}")
            
            return detections, img_with_boxes
            
        except Exception as e:
            print(f"❌ Erro geral na detecção: {str(e)}")
            traceback.print_exc()
            return [], None

    def detect_specific_glasses(self, image):
        """
        Método específico para detecção de óculos de proteção.
        Combina o detector YOLOv8 com métodos adicionais de processamento de imagem
        para melhorar a detecção de óculos de proteção.
        
        Args:
            image: Imagem OpenCV para análise
            
        Returns:
            dict: Resultado da detecção com chaves:
                - detected: bool indicando se óculos foram detectados
                - confidence: nível de confiança da detecção
                - coordinates: coordenadas do objeto detectado [x1, y1, x2, y2]
        """
        result = {
            'detected': False,
            'confidence': 0.0,
            'coordinates': None
        }
        
        try:
            # Tentar detecção com YOLO primeiro
            if self.model and not self.simulation_mode:
                yolo_results = self.model(image, conf=self.confidence_threshold, verbose=False)
                for yolo_result in yolo_results:
                    boxes = yolo_result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Verificar se é a classe de óculos de proteção
                        if self.classes[cls] in ['glasses', 'eyewear', 'óculos', 'oculos']:
                            # Encontrou óculos de proteção
                            result['detected'] = True
                            result['confidence'] = conf
                            # Coordenadas [x1, y1, x2, y2]
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            result['coordinates'] = [x1, y1, x2, y2]
                            return result
            
            # Se não encontrou com YOLO ou está em modo de simulação, tentar detecção baseada em cor
            # Óculos de proteção geralmente têm cores distintas (amarelo, laranja)
            
            # Converter para HSV para melhor detecção de cor
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Faixas de cor para óculos de proteção típicos (amarelo, laranja, azul)
            color_ranges = [
                # Amarelo (comum em óculos de segurança)
                (np.array([20, 100, 100]), np.array([40, 255, 255])),
                # Laranja
                (np.array([5, 100, 100]), np.array([15, 255, 255])),
                # Azul
                (np.array([100, 100, 100]), np.array([130, 255, 255]))
            ]
            
            # Região de interesse - geralmente óculos estão na parte superior do rosto
            height, width = image.shape[:2]
            roi = hsv[0:int(height/2), 0:width]  # Metade superior da imagem
            
            # Procurar por cores de óculos de proteção
            max_ratio = 0
            best_mask = None
            
            for lower, upper in color_ranges:
                mask = cv2.inRange(roi, lower, upper)
                non_zero_pixels = cv2.countNonZero(mask)
                ratio = non_zero_pixels / (roi.shape[0] * roi.shape[1])
                
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_mask = mask
            
            # Se encontrou uma quantidade significativa de pixels da cor esperada
            if max_ratio > 0.005:  # Pelo menos 0.5% da imagem tem a cor de óculos
                result['detected'] = True
                result['confidence'] = min(max_ratio * 10, 0.8)  # Converter para confiança máxima de 0.8
                
                # Encontrar contornos para obter as coordenadas
                if best_mask is not None:
                    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Pegar o maior contorno
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        # Ajustar para coordenadas originais (considerando o ROI)
                        result['coordinates'] = [x, y, x + w, y + h]
            
            return result
                
        except Exception as e:
            print(f"⚠️ Erro na detecção específica de óculos: {str(e)}")
            # Em caso de erro, retornar resultado vazio
            return result

    def get_current_model_classes(self):
        """Retorna a lista de nomes de classes detectáveis pelo modelo atual."""
        print("\n🔍 Obtendo classes do modelo atual...")
        
        # Verificar se temos um modelo atual
        if not self.current_model:
            print("⚠️ Nenhum modelo atual definido")
            return []
            
        # Verificar se temos mapeamento de classes
        if not hasattr(self, 'coco_to_epi') or not self.coco_to_epi:
            print("⚠️ Mapeamento de classes não encontrado")
            return []
            
        # Retornar lista de classes do mapeamento atual
        classes = list(self.coco_to_epi.values())
        print(f"✅ Classes encontradas: {classes}")
        return classes

    def _configure_class_mapping(self):
        """
        Configura o mapeamento de classes para o detector YOLO.
        Se possível, lê as classes do arquivo args.yaml no diretório do modelo.
        Garante que o mapeamento seja sempre um dicionário válido.
        """
        print(f"\n🔍 Configurando mapeamento de classes para o modelo: {self.current_model}")
        
        # Inicializa o dicionário de mapeamento com valores padrão
        self.coco_to_epi = {
            0: "pessoa",
            1: "capacete",
            2: "oculos",
            3: "protetor_auricular",
            4: "mascara",
            5: "luva",
            6: "bota",
            7: "colete"
        }
        
        try:
            # Se não houver modelo atual, mantém o mapeamento padrão
            if not self.current_model:
                print("ℹ️ Nenhum modelo atual definido. Usando mapeamento padrão.")
                return
            
            # Se for modelo YOLO11, usa mapeamento específico
            if 'yolo11' in self.current_model.lower():
                self.coco_to_epi = {
                    0: "pessoa",
                    1: "capacete",
                    2: "colete",
                    3: "luvas",
                    4: "mascara",
                    5: "oculos",
                    6: "protetor_auricular"
                }
                print(f"✅ Usando mapeamento específico para YOLO11: {self.coco_to_epi}")
                return
            
            # Tenta encontrar o arquivo args.yaml
            model_info = self.models_cache.get(self.current_model)
            if not model_info:
                print("⚠️ Informações do modelo não encontradas no cache.")
                return
                
            model_path = model_info.get('path')
            if not model_path:
                print("⚠️ Caminho do modelo não encontrado no cache.")
                return
                
            model_dir = os.path.dirname(model_path)
            args_path = os.path.join(model_dir, "args.yaml")
            
            if not os.path.exists(args_path):
                print(f"ℹ️ Arquivo args.yaml não encontrado em: {args_path}")
                print(f"ℹ️ Mantendo mapeamento padrão: {self.coco_to_epi}")
                return
                
            print(f"📄 Encontrado arquivo args.yaml em: {args_path}")
            with open(args_path, 'r') as f:
                args_data = yaml.safe_load(f)
            
            if not args_data or 'names' not in args_data:
                print("⚠️ Arquivo args.yaml não contém informações de classes.")
                return
                
            names = args_data['names']
            
            # Garante que o mapeamento seja sempre um dicionário
            if isinstance(names, dict):
                # Converte chaves para inteiros
                self.coco_to_epi = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, list):
                # Converte lista para dicionário
                self.coco_to_epi = {i: str(name) for i, name in enumerate(names)}
            else:
                print(f"⚠️ Formato inesperado de classes em args.yaml: {type(names)}")
                return
                
            print(f"✅ Classes carregadas do args.yaml: {self.coco_to_epi}")
            
        except Exception as e:
            print(f"⚠️ Erro ao configurar classes: {str(e)}")
            print(f"ℹ️ Mantendo mapeamento padrão: {self.coco_to_epi}")
            
        finally:
            # Garante que coco_to_epi seja sempre um dicionário
            if not isinstance(self.coco_to_epi, dict):
                print("⚠️ Mapeamento inválido detectado. Restaurando padrão.")
                self.coco_to_epi = {
                    0: "pessoa",
                    1: "capacete",
                    2: "oculos",
                    3: "protetor_auricular",
                    4: "mascara",
                    5: "luva",
                    6: "bota",
                    7: "colete"
                }

    def _normalize_class_name(self, class_name):
        """
        Normaliza o nome da classe para garantir consistência.
        Converte variações em inglês/português para um formato padrão.
        """
        # Converter para minúsculas e remover acentos
        normalized = class_name.lower().strip()
        
        # Mapeamento de normalização
        normalization_map = {
            # Inglês -> Português
            'person': 'pessoa',
            'helmet': 'capacete',
            'glasses': 'oculos',
            'eyewear': 'oculos',
            'goggles': 'oculos',
            'mask': 'mascara',
            'gloves': 'luvas',
            'glove': 'luvas',
            'boots': 'botas',
            'boot': 'botas',
            'vest': 'colete',
            'ear_protection': 'protetor_auricular',
            'ear_protector': 'protetor_auricular',
            
            # Normalizações em português
            'óculos': 'oculos',
            'máscara': 'mascara',
            'luva': 'luvas',
            'bota': 'botas',
            'protetor auricular': 'protetor_auricular'
        }
        
        # Retornar versão normalizada ou o original se não houver mapeamento
        return normalization_map.get(normalized, normalized)

# Verificar configuração do dataset para o YOLO
def setup_dataset_directories():
    """Configura os diretórios do dataset para compatibilidade com YOLO."""
    try:
        import os
        import shutil
        
        # Caminho onde o YOLO procura os datasets por padrão (agora usando o diretório do projeto)
        datasets_dir = os.path.join(os.getcwd(), 'datasets')
        target_dataset_dir = os.path.join(datasets_dir, 'dataset_treinamento')
        
        # Caminhos locais
        local_dataset_dir = os.path.abspath('dataset_treinamento')
        
        print(f"🔍 Verificando diretórios para treinamento:")
        print(f"   - YOLO procura em: {datasets_dir}")
        print(f"   - Nosso dataset está em: {local_dataset_dir}")
        
        # Verificar se existe o diretório local
        if not os.path.exists(local_dataset_dir):
            os.makedirs(local_dataset_dir, exist_ok=True)
            os.makedirs(os.path.join(local_dataset_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(local_dataset_dir, 'labels'), exist_ok=True)
            print(f"✅ Diretórios locais criados: {local_dataset_dir}")
        
        # Verificar se o diretório esperado pelo YOLO existe
        os.makedirs(datasets_dir, exist_ok=True)
        print(f"✅ Diretório de datasets do YOLO criado: {datasets_dir}")
        
        # Copiar arquivos de dataset_treinamento para o local esperado pelo YOLO
        if os.path.exists(local_dataset_dir) and os.listdir(local_dataset_dir):
            print(f"🔄 Copiando dataset do diretório local para o diretório YOLO...")
            
            # Garantir que o diretório de destino exista
            os.makedirs(target_dataset_dir, exist_ok=True)
            os.makedirs(os.path.join(target_dataset_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(target_dataset_dir, 'labels'), exist_ok=True)
            
            # Copiar imagens
            local_images_dir = os.path.join(local_dataset_dir, 'images')
            target_images_dir = os.path.join(target_dataset_dir, 'images')
            if os.path.exists(local_images_dir):
                for file in os.listdir(local_images_dir):
                    source = os.path.join(local_images_dir, file)
                    dest = os.path.join(target_images_dir, file)
                    try:
                        if not os.path.exists(dest) or os.path.getmtime(source) > os.path.getmtime(dest):
                            shutil.copy2(source, dest)
                    except Exception as e:
                        print(f"⚠️ Erro ao copiar {file}: {str(e)}")
                
                print(f"✅ Imagens copiadas para {target_images_dir}")
            
            # Copiar labels
            local_labels_dir = os.path.join(local_dataset_dir, 'labels')
            target_labels_dir = os.path.join(target_dataset_dir, 'labels')
            if os.path.exists(local_labels_dir):
                for file in os.listdir(local_labels_dir):
                    source = os.path.join(local_labels_dir, file)
                    dest = os.path.join(target_labels_dir, file)
                    try:
                        if not os.path.exists(dest) or os.path.getmtime(source) > os.path.getmtime(dest):
                            shutil.copy2(source, dest)
                    except Exception as e:
                        print(f"⚠️ Erro ao copiar {file}: {str(e)}")
                
                print(f"✅ Labels copiados para {target_labels_dir}")
            
            # Criar arquivo dataset.yaml para o YOLO
            try:
                yaml_content = """
# Dataset configuration for YOLO training
train: images/  # Train images (relative to dataset directory)
val: images/    # Validation images (using same as train for now)

# Class names
names:
  0: pessoa
  1: capacete
  2: oculos
  3: protetor_auricular
  4: mascara
  5: luva
  6: bota
  7: colete
"""
                # Salvar o arquivo no diretório local e no diretório YOLO
                local_yaml_path = os.path.join(local_dataset_dir, 'dataset.yaml')
                target_yaml_path = os.path.join(target_dataset_dir, 'dataset.yaml')
                
                with open(local_yaml_path, 'w') as f:
                    f.write(yaml_content)
                
                with open(target_yaml_path, 'w') as f:
                    f.write(yaml_content)
                
                print(f"✅ Arquivo dataset.yaml criado em ambos os diretórios")
            except Exception as e:
                print(f"⚠️ Erro ao criar arquivo dataset.yaml: {str(e)}")
            
            print(f"✅ Dataset sincronizado com o diretório esperado pelo YOLO")
            
            return target_dataset_dir
        else:
            print(f"⚠️ Diretório de dataset local vazio ou inexistente. Nada a copiar.")
            return local_dataset_dir
    
    except Exception as e:
        print(f"❌ Erro na configuração dos diretórios do dataset: {str(e)}")
        return None

# Executar a configuração de diretórios
dataset_path = setup_dataset_directories()

# Garantir que o modelo padrão ainda exista após a descoberta
DEFAULT_MODEL = 'epi_detector2' # Ou o nome do seu melhor modelo treinado
if DEFAULT_MODEL not in YOLO_MODELS:
     print(f"⚠️ Modelo padrão '{DEFAULT_MODEL}' não encontrado após descoberta. Verificando alternativas...")
     # Tentar encontrar qualquer modelo 'epi_detector*'
     epi_models = [m for m in YOLO_MODELS if m.startswith('epi_detector')]
     if epi_models:
         DEFAULT_MODEL = sorted(epi_models)[-1] # Pegar o mais recente (ex: epi_detector3)
         print(f"   -> Usando modelo alternativo: '{DEFAULT_MODEL}'")
     else:
         # Se nenhum epi_detector for encontrado, usar yolov8n como fallback
         DEFAULT_MODEL = 'yolov8n'
         print(f"   -> Nenhum modelo 'epi_detector' encontrado. Usando fallback: '{DEFAULT_MODEL}'")

# ... (Restante das rotas, incluindo /pre_anotar_imagem)

def discover_local_models():
    """
    Descobre modelos disponíveis localmente e retorna um dicionário com nome: caminho.
    Prioriza modelos na pasta models/ e reverte para os modelos padrão do YOLO se nenhum for encontrado.
    O dicionário retornado tem o formato {nome_modelo: 'local:/caminho/absoluto/para/modelo.pt'}
    """
    # Resultado: dicionário de modelos encontrados
    local_models = {}
    
    # Verificar se a pasta models/ existe, caso contrário, criá-la
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # PARTE 1: Copiar modelos .pt da raiz para models/ se existirem
    print("\nVerificando arquivos .pt na raiz do projeto...")
    root_pt_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.pt') and os.path.isfile(os.path.join(os.getcwd(), f))]
    
    if root_pt_files:
        print(f"Encontrados {len(root_pt_files)} arquivos .pt na raiz:")
        for pt_file in root_pt_files:
            source_path = os.path.join(os.getcwd(), pt_file)
            target_path = os.path.join(models_dir, pt_file)
            print(f"   - {pt_file} ({os.path.getsize(source_path) / (1024*1024):.1f} MB)")
            
            # Copiar arquivo se não existir em models/ ou se for mais novo
            if not os.path.exists(target_path) or os.path.getmtime(source_path) > os.path.getmtime(target_path):
                print(f"      Copiando para models/...")
                shutil.copy2(source_path, target_path)
    else:
        print("Nenhum arquivo .pt encontrado na raiz.")
    
    # PARTE 2: Verificar modelos em models/ (incluindo os recém-copiados)
    print("\nVerificando modelos na pasta models/...")
    models_files = os.listdir(models_dir)
    valid_models = []
    
    min_size_mb = 1  # Tamanho mínimo em MB para considerar um arquivo como modelo válido
    
    for filename in models_files:
        if filename.endswith('.pt'):
            filepath = os.path.join(models_dir, filename)
            filesize_mb = os.path.getsize(filepath) / (1024*1024)
            
            if filesize_mb > min_size_mb:
                model_name = os.path.splitext(filename)[0]  # Remove a extensão .pt
                valid_models.append((model_name, filepath, filesize_mb))
                print(f"   ✅ {model_name}: {filepath} ({filesize_mb:.1f} MB)")
            else:
                print(f"   ❌ {filename}: muito pequeno ({filesize_mb:.1f} MB < {min_size_mb} MB)")
    
    # PARTE 3: Verificar na pasta runs/detect/{model}/weights/ para modelos treinados
    runs_dir = os.path.join(os.getcwd(), 'runs', 'detect')
    if os.path.exists(runs_dir):
        print("\nVerificando modelos na pasta runs/detect/...")
        for model_dir in os.listdir(runs_dir):
            weights_dir = os.path.join(runs_dir, model_dir, 'weights')
            best_pt = os.path.join(weights_dir, 'best.pt')
            
            if os.path.exists(best_pt) and os.path.isfile(best_pt):
                filesize_mb = os.path.getsize(best_pt) / (1024*1024)
                if filesize_mb > min_size_mb:
                    # Verificar se já temos este modelo
                    if model_dir in [name for name, _, _ in valid_models]:
                        print(f"   ⚠️ {model_dir}: já encontrado em models/")
                    else:
                        valid_models.append((model_dir, best_pt, filesize_mb))
                        print(f"   ✅ {model_dir}: {best_pt} ({filesize_mb:.1f} MB)")
                    
                    # Tentar copiar para models/ para futura descoberta
                    target_path = os.path.join(models_dir, f"{model_dir}.pt")
                    if not os.path.exists(target_path):
                        print(f"      Copiando para models/{model_dir}.pt...")
                        try:
                            shutil.copy2(best_pt, target_path)
                        except Exception as e:
                            print(f"      ❌ Erro ao copiar: {str(e)}")
    else:
        print("\nPasta runs/detect/ não encontrada. Pulando busca por modelos treinados.")
    
    # PARTE 4: Montar o dicionário de resultado
    if valid_models:
        print(f"\nEncontrados {len(valid_models)} modelos válidos:")
        for model_name, model_path, filesize_mb in valid_models:
            local_models[model_name] = f"local:{model_path}"
            print(f"   - {model_name}: {model_path} ({filesize_mb:.1f} MB)")
    
    # Retornar os modelos encontrados
    return local_models

def initialize_detector(preferred_model=None):
    """Inicializa o detector de EPI com o modelo preferido ou o primeiro disponível."""
    global DETECTOR
    
    print(f"\n📋 INITIALIZE_DETECTOR: Modelos disponíveis ({len(YOLO_MODELS)}):")
    for model_name, model_path in YOLO_MODELS.items():
        print(f"   - {model_name}: {model_path}")
    
    if not YOLO_MODELS:
        print("❌ ERRO: Nenhum modelo disponível para inicialização do detector!")
        return False
    
    if DETECTOR is not None:
        print(f"ℹ️ Detector já inicializado com modelo: {getattr(DETECTOR, 'current_model', 'desconhecido')}")
        return True
    
    # Cria uma nova instância do detector
    DETECTOR = YoloEPIDetector()
    
    # Define a ordem de preferência para os modelos
    preferred_models = [
        preferred_model,  # Primeiro o modelo solicitado (se houver)
        "epi_detector3",  # Depois nossos modelos customizados por ordem de prioridade
        "epi_detector2",
        "epi_detector",
        "yolov8n"         # Por último um modelo padrão pequeno
    ]
    
    # Tenta carregar cada modelo na ordem de preferência
    for model_name in preferred_models:
        if model_name and model_name in YOLO_MODELS:
            print(f"🔍 Tentando carregar modelo '{model_name}'...")
            if DETECTOR.load_model(model_name):
                print(f"✅ Modelo '{model_name}' carregado com sucesso!")
                return True
            else:
                print(f"⚠️ Falha ao carregar modelo '{model_name}'. Tentando próximo...")
    
    # Se nenhum modelo preferido funcionou, tenta qualquer modelo disponível
    print("⚠️ Nenhum modelo preferido pôde ser carregado. Tentando qualquer modelo disponível...")
    for model_name in YOLO_MODELS:
        if model_name not in preferred_models:
            print(f"🔍 Tentando carregar modelo alternativo '{model_name}'...")
            if DETECTOR.load_model(model_name):
                print(f"✅ Modelo alternativo '{model_name}' carregado com sucesso!")
                return True
            else:
                print(f"⚠️ Falha ao carregar modelo alternativo '{model_name}'.")
    
    print("❌ ERRO CRÍTICO: Nenhum modelo pôde ser carregado! Detector não funcional.")
    return False

@app.route('/debug', methods=['GET'])
def debug_system():
    """Rota de diagnóstico para depurar o sistema."""
    global detector, YOLO_MODELS
    
    results = {
        "status": "success",
        "detector_status": "OK" if detector else "Não inicializado",
        "current_model": detector.current_model if detector else None,
        "actions_taken": [],
        "yolo_models": list(YOLO_MODELS.keys()),
        "error_details": []
    }
    
    # Verificar se temos modelos disponíveis
    if not YOLO_MODELS:
        results["actions_taken"].append("Redescobrindo modelos...")
        discovered = discover_local_models()
        YOLO_MODELS.update(discovered)
        results["yolo_models"] = list(YOLO_MODELS.keys())
        
        if not YOLO_MODELS:
            results["actions_taken"].append("Nenhum modelo local encontrado. Adicionando modelos padrão YOLO...")
            YOLO_MODELS.update(YOLO_MODELS_BASE)
            results["yolo_models"] = list(YOLO_MODELS.keys())
    
    # Verificar se o detector está inicializado
    if detector is None:
        results["actions_taken"].append("Detector não inicializado. Criando nova instância...")
        detector = YoloEPIDetector()
    
    # Tentar carregar um modelo
    if not detector.current_model:
        results["actions_taken"].append("Detector não tem modelo atual. Tentando carregar...")
        
        for model_name in ['epi_detector3', 'epi_detector2', 'epi_detector', 'yolov8n', 'yolo11n']:
            if model_name in YOLO_MODELS:
                results["actions_taken"].append(f"Tentando carregar modelo: {model_name}")
                
                try:
                    model_obj = detector.load_model(model_to_load=model_name)
                    if model_obj:
                        detector.current_model = model_name
                        results["actions_taken"].append(f"✅ Modelo {model_name} carregado com sucesso!")
                        results["current_model"] = model_name
                        break
                    else:
                        results["actions_taken"].append(f"❌ Falha ao carregar modelo {model_name}")
                except Exception as e:
                    error_details = {
                        "model": model_name,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    results["error_details"].append(error_details)
                    results["actions_taken"].append(f"❌ Erro ao carregar modelo {model_name}: {str(e)}")
    
    # Verificar status final
    if detector and detector.current_model:
        results["detector_status"] = f"OK - Modelo {detector.current_model} ativo"
        
        try:
            classes = detector.get_current_model_classes()
            results["classes"] = classes
            results["actions_taken"].append(f"Classes do modelo: {classes}")
        except Exception as e:
            results["actions_taken"].append(f"❌ Erro ao obter classes: {str(e)}")
    else:
        results["detector_status"] = "FALHA - Nenhum modelo ativo"
    
    return jsonify(results)

@app.route('/reset', methods=['GET'])
def reset_system():
    global detector, YOLO_MODELS
    detector = None
    YOLO_MODELS.clear()
    return jsonify({"status": "success", "message": "Sistema reiniciado com sucesso"})

@app.route('/change_model', methods=['POST'])
def change_model():
    """
    Rota para alterar o modelo atual via método POST.
    Recebe JSON com o parâmetro 'model'.
    """
    try:
        # Verificar se recebemos um JSON válido
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Dados JSON não fornecidos'
            }), 400
        
        # Obter o nome do modelo a ser carregado
        data = request.get_json()
        model_name = data.get('model')
        
        if not model_name:
            return jsonify({
                'status': 'error',
                'message': 'Nome do modelo não fornecido'
            }), 400
        
        print(f"\n🔄 Solicitação para mudar modelo para: {model_name}")
        
        # Verificar se o modelo está disponível
        if model_name not in YOLO_MODELS:
            return jsonify({
                'status': 'error',
                'message': f'Modelo {model_name} não encontrado nos modelos disponíveis',
                'available_models': list(YOLO_MODELS.keys())
            }), 404
        
        global detector
        if not detector:
            print("⚠️ Detector não inicializado. Inicializando...")
            detector = YoloEPIDetector()
        
        # Guardar o modelo atual para caso algo dê errado
        old_model = detector.current_model
        
        # Tentar carregar o novo modelo
        detector.current_model = model_name
        model = detector.load_model()
        
        if not model:
            # Restaurar modelo anterior em caso de falha
            detector.current_model = old_model
            return jsonify({
                'status': 'error',
                'message': f'Falha ao carregar modelo {model_name}',
                'current_model': old_model
            }), 500
        
        # Modelo carregado com sucesso
        print(f"✅ Modelo alterado para {model_name}")
        
        # Obter classes do novo modelo
        try:
            classes = detector.get_current_model_classes()
            print(f"📋 Classes disponíveis no modelo {model_name}: {classes}")
        except Exception as e:
            print(f"⚠️ Erro ao obter classes: {str(e)}")
            classes = []
        
        return jsonify({
            'status': 'success',
            'message': f'Modelo alterado para {model_name}',
            'current_model': model_name,
            'classes': classes
        })
        
    except Exception as e:
        print(f"❌ Erro ao alterar modelo: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Erro interno: {str(e)}'
        }), 500

@app.route('/debug_analyze', methods=['POST'])
def debug_analyze():
    """Rota simplificada para diagnosticar problemas na análise de imagens."""
    try:
        print("\n🔬 INICIANDO DIAGNÓSTICO DE ANÁLISE")
        
        # Verificar detector
        if not detector:
            print("❌ Detector não inicializado ou é None")
            return jsonify({
                'status': 'error',
                'message': 'Detector não inicializado',
                'step': 'detector_check'
            }), 500
        
        print(f"✅ Detector inicializado: {type(detector)}")
        print(f"✅ Modelo atual: {detector.current_model}")
        
        # Verificar arquivo de imagem
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Nenhuma imagem enviada',
                'step': 'image_check'
            }), 400
        
        # Salvar imagem
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Nome de arquivo vazio',
                'step': 'filename_check'
            }), 400
        
        # Gerar um nome único para o arquivo
        filename = secure_filename(image_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"debug_{timestamp}_{filename}"
        
        # Salvar o arquivo em disco
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image_file.save(image_path)
        print(f"✅ Imagem salva em: {image_path}")
        
        # Verificar classes selecionadas
        selected_classes = []
        try:
            if 'selected_classes' in request.form:
                selected_classes = json.loads(request.form['selected_classes'])
                print(f"✅ Classes selecionadas: {selected_classes}")
        except Exception as e:
            print(f"⚠️ Erro ao processar classes: {str(e)}")
            print(f"⚠️ Conteúdo raw: {request.form.get('selected_classes', 'VAZIO')}")
        
        # Tentar detectar objetos na imagem (versão simplificada)
        try:
            print("⏳ Iniciando detecção...")
            detections = detector.detect(image_path)
            print(f"✅ Detecção concluída. Encontradas {len(detections)} detecções")
            
            # Se não há detecções, retornar uma resposta simples
            if not detections:
                return jsonify({
                    'status': 'success',
                    'message': 'Nenhuma detecção encontrada',
                    'image_path': unique_filename,
                    'detections': []
                })
            
            # Filtrar por classes se necessário
            if selected_classes:
                filtered = [d for d in detections if d.get('class_name') in selected_classes]
                print(f"✅ Filtro aplicado. Mantidas {len(filtered)} detecções")
            else:
                filtered = detections
            
            # Retornar resultados básicos sem processamento adicional
            return jsonify({
                'status': 'success',
                'message': 'Análise de diagnóstico concluída',
                'image_path': unique_filename,
                'detections': filtered
            })
            
        except Exception as e:
            print(f"❌ ERRO DURANTE DETECÇÃO: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'message': f'Erro durante detecção: {str(e)}',
                'traceback': traceback.format_exc(),
                'step': 'detection'
            }), 500
        
    except Exception as e:
        print(f"❌ ERRO GERAL: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Erro geral: {str(e)}',
            'traceback': traceback.format_exc(),
            'step': 'general'
        }), 500

# --- INÍCIO: Descoberta de Modelos e Preenchimento de YOLO_MODELS ---
print("\n🔍 Descobrindo modelos disponíveis (execução global)...")
discovered_models = discover_local_models()

# Atualizar o dicionário YOLO_MODELS
if discovered_models:
    print(f"\n✅ Atualizando YOLO_MODELS com {len(discovered_models)} modelos (execução global)")
    YOLO_MODELS.clear()  # Limpar dicionário anterior
    YOLO_MODELS.update(discovered_models)
else:
    print("\n⚠️ Nenhum modelo encontrado na pasta models/ (execução global)")
    # Adicionar modelos base do YOLO se nenhum modelo local for encontrado
    YOLO_MODELS.update(YOLO_MODELS_BASE)
    print(f"✅ Adicionados {len(YOLO_MODELS_BASE)} modelos base do YOLO (execução global)")

print(f"\n📋 Lista final de modelos disponíveis ({len(YOLO_MODELS)}) (execução global):")
for model_name in sorted(YOLO_MODELS.keys()):
    model_path = YOLO_MODELS[model_name]
    print(f"   - {model_name}: {model_path}")
# --- FIM: Descoberta de Modelos e Preenchimento de YOLO_MODELS ---

# Resetar o detector para garantir inicialização limpa
DETECTOR = None
print("\n🚀 Iniciando initialize_detector() com modelos disponíveis...")
status = initialize_detector()
print(f"🏁 Finalizado initialize_detector() - Status: {'✅ Sucesso' if status else '❌ Falha'}")

# Verificar se o detector está funcionando
if not status or DETECTOR is None:
    print("⚠️ DETECTOR não foi inicializado corretamente. Tentando novamente...")
    # Tentar com um modelo específico se estiver disponível
    preferred_model = None
    for model_name in ["epi_detector3", "epi_detector2", "epi_detector", "yolov8n"]:
        if model_name in YOLO_MODELS:
            preferred_model = model_name
            print(f"   Modelo preferido encontrado: {model_name}")
            break
    
    if preferred_model:
        print(f"   Tentando especificamente com o modelo: {preferred_model}")
        status = initialize_detector(preferred_model)
    else:
        print("   Nenhum modelo preferido encontrado. Tentando qualquer modelo disponível...")
        status = initialize_detector()
    
    print(f"   Resultado da segunda tentativa: {'✅ Sucesso' if status else '❌ Falha'}")
else:
    print(f"✅ DETECTOR inicializado com modelo: {DETECTOR.current_model if hasattr(DETECTOR, 'current_model') else 'desconhecido'}")
    print(f"   Classes disponíveis: {len(DETECTOR.coco_to_epi) if hasattr(DETECTOR, 'coco_to_epi') else 'desconhecido'}")

setup_dataset_directories()

if __name__ == '__main__':
    print(f"🚀 Iniciando servidor na porta 5000...")
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {str(e)}")

@app.route('/api/dataset/clear', methods=['POST'])
def api_clear_dataset():
    """Remove todas as imagens e anotações do dataset atual."""
    try:
        # Verificar diretórios
        images_dir = os.path.join('dataset_treinamento', 'images')
        labels_dir = os.path.join('dataset_treinamento', 'labels')
        
        # Contar arquivos antes da limpeza
        num_images = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0
        num_labels = len(os.listdir(labels_dir)) if os.path.exists(labels_dir) else 0
        
        # Remover arquivos de imagem
        if os.path.exists(images_dir):
            for file in os.listdir(images_dir):
                file_path = os.path.join(images_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"❌ Erro ao remover arquivo {file_path}: {str(e)}")
        
        # Remover arquivos de anotação
        if os.path.exists(labels_dir):
            for file in os.listdir(labels_dir):
                file_path = os.path.join(labels_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"❌ Erro ao remover arquivo {file_path}: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': f'Dataset limpo com sucesso. Removidas {num_images} imagens e {num_labels} anotações.',
            'removed_images': num_images,
            'removed_labels': num_labels
        })
    except Exception as e:
        print(f"❌ Erro ao limpar dataset: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500