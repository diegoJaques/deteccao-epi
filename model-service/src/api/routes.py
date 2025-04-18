from flask import Blueprint, request, jsonify
from ..core.model_handler import ModelHandler

api = Blueprint('api', __name__)
model_handler = ModelHandler()

@api.route('/models', methods=['GET'])
def list_models():
    """Lista todos os modelos disponíveis."""
    try:
        models = model_handler.list_models()
        return jsonify({
            'status': 'success',
            'models': models
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/models/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """Obtém informações sobre um modelo específico."""
    try:
        info = model_handler.get_model_info(model_name)
        return jsonify(info)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/models/<model_name>', methods=['POST'])
def download_model(model_name):
    """Baixa um modelo específico."""
    try:
        result = model_handler.download_model(model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/models/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """Remove um modelo específico."""
    try:
        result = model_handler.delete_model(model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/models/train', methods=['POST'])
def train_model():
    """Inicia o treinamento de um novo modelo."""
    try:
        # Verificar se foi enviado um arquivo de dataset
        if 'dataset' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Nenhum dataset enviado'
            }), 400
            
        dataset = request.files['dataset']
        if dataset.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'Nenhum arquivo selecionado'
            }), 400
            
        # Obter parâmetros de treinamento
        params = request.form.get('params', '{}')
        params = eval(params)  # Converter string para dicionário
        
        # Iniciar treinamento
        # TODO: Implementar lógica de treinamento
        return jsonify({
            'status': 'success',
            'message': 'Treinamento iniciado'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 