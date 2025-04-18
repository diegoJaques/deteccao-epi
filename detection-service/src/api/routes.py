from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
from ..core.detector import EPIDetector

api = Blueprint('api', __name__)
detector = EPIDetector()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api.route('/models', methods=['GET'])
def get_available_models():
    """Lista todos os modelos disponíveis."""
    try:
        return jsonify({
            'status': 'success',
            'models': list(detector.available_models.keys()),
            'current_model': detector.current_model
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/models/<model_name>', methods=['POST'])
def set_model(model_name):
    """Define o modelo a ser usado."""
    try:
        success = detector.load_model(model_name)
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Modelo {model_name} carregado com sucesso'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Falha ao carregar modelo {model_name}'
            }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/detect', methods=['POST'])
def detect_epis():
    """Detecta EPIs em uma imagem."""
    try:
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Nenhuma imagem enviada'
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Nenhum arquivo selecionado'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': 'Tipo de arquivo não permitido'
            }), 400

        # Ler imagem
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Obter EPIs selecionados
        selected_epis = request.form.get('selected_epis', '[]')
        selected_epis = eval(selected_epis)  # Converter string para lista

        # Realizar detecção
        result = detector.detect_epis(image, selected_epis)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/detect/specific', methods=['POST'])
def detect_specific_epi():
    """Detecta um EPI específico em uma imagem."""
    try:
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Nenhuma imagem enviada'
            }), 400

        file = request.files['image']
        epi_type = request.form.get('epi_type')

        if not epi_type:
            return jsonify({
                'status': 'error',
                'message': 'Tipo de EPI não especificado'
            }), 400

        # Ler imagem
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Realizar detecção específica
        result = detector.detect_specific_epi(image, epi_type)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 