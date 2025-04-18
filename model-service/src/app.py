from flask import Flask
from api.routes import api
import os
from config import get_config

def create_app():
    app = Flask(__name__)
    
    # Carregar configurações
    config = get_config()
    app.config.from_object(config)
    
    # Criar diretórios necessários
    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
    os.makedirs(app.config['DATASET_DIR'], exist_ok=True)
    
    # Registrar blueprint da API
    app.register_blueprint(api, url_prefix='/api')
    
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'model-service'}
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001) 