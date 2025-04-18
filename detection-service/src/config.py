import os

class Config:
    """Configurações base."""
    DEBUG = False
    TESTING = False
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max-limit
    
    # URLs dos outros serviços
    MODEL_SERVICE_URL = os.getenv('MODEL_SERVICE_URL', 'http://model-service:5001')
    STATS_SERVICE_URL = os.getenv('STATS_SERVICE_URL', 'http://stats-service:5002')
    NOTIFICATION_SERVICE_URL = os.getenv('NOTIFICATION_SERVICE_URL', 'http://notification-service:5003')

class DevelopmentConfig(Config):
    """Configurações de desenvolvimento."""
    DEBUG = True
    
class ProductionConfig(Config):
    """Configurações de produção."""
    # Configurações específicas de produção
    pass

class TestingConfig(Config):
    """Configurações de teste."""
    TESTING = True
    
# Dicionário de configurações
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Função para obter a configuração atual
def get_config():
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default']) 