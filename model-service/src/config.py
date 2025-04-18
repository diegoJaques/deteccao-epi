import os

class Config:
    """Configurações base."""
    DEBUG = False
    TESTING = False
    
    # Diretórios
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
    
    # Limites
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max-limit
    
    # URLs dos outros serviços
    DETECTION_SERVICE_URL = os.getenv('DETECTION_SERVICE_URL', 'http://detection-service:5000')
    STATS_SERVICE_URL = os.getenv('STATS_SERVICE_URL', 'http://stats-service:5002')
    
    # Configurações de treinamento
    TRAINING_BATCH_SIZE = 16
    TRAINING_EPOCHS = 100
    TRAINING_IMG_SIZE = 640
    TRAINING_WORKERS = 4

class DevelopmentConfig(Config):
    """Configurações de desenvolvimento."""
    DEBUG = True
    
    # Reduzir épocas para desenvolvimento
    TRAINING_EPOCHS = 10
    
class ProductionConfig(Config):
    """Configurações de produção."""
    # Aumentar limites para produção
    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB
    TRAINING_BATCH_SIZE = 32
    TRAINING_WORKERS = 8
    
class TestingConfig(Config):
    """Configurações de teste."""
    TESTING = True
    
    # Configurações mínimas para testes
    TRAINING_EPOCHS = 1
    TRAINING_BATCH_SIZE = 2
    
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