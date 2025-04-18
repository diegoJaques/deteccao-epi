"""
Script para configurar corretamente o Ultralytics para treinamento.
Execução: py configurar_ultralytics.py
"""

import os
import json
import sys
import yaml
from pathlib import Path
import subprocess

def main():
    print("Configurando Ultralytics para treinamento")
    print("========================================")
    
    # Verificar se ultralytics está instalado
    try:
        import ultralytics
        print(f"Ultralytics versão {ultralytics.__version__} encontrado")
    except ImportError:
        print("Ultralytics não está instalado. Tentando instalar...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics", "--upgrade"])
        
        # Verificar novamente
        try:
            import ultralytics
            print(f"Ultralytics instalado com sucesso (versão {ultralytics.__version__})")
        except ImportError:
            print("ERRO: Falha ao instalar ultralytics. Por favor, instale manualmente com:")
            print("pip install ultralytics --upgrade")
            return False
    
    # Configurar o diretório de datasets
    print("\nConfigurando diretório de datasets...")
    
    # Obter o diretório atual como base para datasets
    base_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, "dataset_treinamento")
    
    # Garantir que as pastas existem
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)
    
    # Configurar o arquivo settings.json do Ultralytics
    settings_dir = os.path.join(os.getenv('APPDATA'), 'Ultralytics')
    os.makedirs(settings_dir, exist_ok=True)
    
    settings_file = os.path.join(settings_dir, 'settings.json')
    settings = {"datasets_dir": base_dir}
    
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)
    
    print(f"Configurações salvas em: {settings_file}")
    print(f"Diretório base para datasets: {base_dir}")
    
    # Atualizar o arquivo de configuração do dataset
    yaml_path = os.path.join(base_dir, "dataset_config.yaml")
    
    # Verificar se o arquivo existe, se não, criar
    if not os.path.exists(yaml_path):
        config = {
            "path": "./dataset_treinamento",
            "train": "images",
            "val": "images",
            "names": {
                0: "pessoa",
                1: "capacete",
                2: "oculos_protecao",
                3: "mascara",
                4: "luvas",
                5: "colete"
            }
        }
        
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(config, f)
            
        print(f"Arquivo de configuração criado: {yaml_path}")
    else:
        # Atualizar arquivo existente
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Garantir configurações corretas
        config["path"] = "./dataset_treinamento"
        config["train"] = "images"
        config["val"] = "images"
        
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(config, f)
            
        print(f"Arquivo de configuração atualizado: {yaml_path}")
    
    print("\nConfiguração concluída com sucesso!")
    print("Agora você pode iniciar o treinamento com: py novo_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nTudo pronto! Pressione Enter para sair...")
    else:
        print("\nHouve erros na configuração. Pressione Enter para sair...")
    
    input() 