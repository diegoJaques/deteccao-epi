from flask import Flask
from api.routes import api
import os

def create_app():
    app = Flask(__name__)
    
    # Configurações
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Registrar blueprint da API
    app.register_blueprint(api, url_prefix='/api')
    
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'detection-service'}
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000) 