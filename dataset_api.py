"""
API para gerenciamento do dataset.

Para usar este arquivo:
1. Importe-o no app.py: `from dataset_api import register_dataset_api`
2. Adicione a seguinte linha após a criação da app: `register_dataset_api(app)`
"""

import os
from flask import jsonify

def register_dataset_api(app):
    """Registra os endpoints da API de dataset."""
    
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