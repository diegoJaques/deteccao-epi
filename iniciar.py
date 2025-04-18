#!/usr/bin/env python3
import os
import sys
import pkg_resources
import subprocess
import webbrowser
from pathlib import Path
import socket
import traceback

def verificar_dependencias():
    """Verifica se todas as dependências estão instaladas."""
    required = {'opencv-python', 'numpy', 'tensorflow', 'tensorflow-hub', 'flask', 'pillow', 'python-dotenv'}
    instaladas = {pkg.key for pkg in pkg_resources.working_set}
    faltantes = required - instaladas
    
    if faltantes:
        print(f"As seguintes dependências estão faltando: {faltantes}")
        print("Instalando dependências...")
        
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Dependências instaladas com sucesso!")
    else:
        print("Todas as dependências estão instaladas!")

def verificar_diretorio():
    """Verifica se os diretórios necessários existem."""
    diretorios = ['uploads', 'templates']
    
    for diretorio in diretorios:
        Path(diretorio).mkdir(exist_ok=True)
        print(f"Diretório {diretorio} verificado.")

def verificar_env():
    """Verifica se o arquivo .env existe."""
    if not os.path.exists('.env'):
        print("Criando arquivo .env padrão...")
        with open('.env', 'w') as f:
            f.write("""# Configurações de Email
NOTIFICATION_EMAIL=admin@empresa.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=seu_email@gmail.com
SMTP_PASSWORD=sua_senha_de_app

# Configurações do Detector
CONFIDENCE_THRESHOLD=0.5""")
        
        print("Arquivo .env criado. Por favor, edite-o com suas configurações de email.")
    else:
        print("Arquivo .env encontrado.")

def iniciar_sistema():
    """Inicia o sistema."""
    print("\n" + "="*50)
    print("   SISTEMA DE MONITORAMENTO DE EPIs")
    print("="*50)
    
    print("\nVerificando dependências...")
    verificar_dependencias()
    
    print("\nVerificando diretórios...")
    verificar_diretorio()
    
    print("\nVerificando arquivo .env...")
    verificar_env()
    
    print("\nIniciando o servidor...")
    
    # Obter IP da máquina para facilitar acesso pela rede
    def get_ip():
        try:
            # Criar socket para determinar qual interface está conectada à rede
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"  # fallback para localhost
    
    host_ip = get_ip()
    port = 5000  # Usar a porta padrão 5000
    
    print(f"\n✅ Servidor iniciando...")
    print(f"🌐 Acesse a aplicação em:")
    print(f"   - Local: http://127.0.0.1:{port}")
    print(f"   - Rede: http://{host_ip}:{port}")
    print("\n⚠️ Para acessar pela rede, certifique-se de permitir")
    print("   a conexão no firewall, se necessário.")
    print("="*50 + "\n")
    
    # Iniciar o servidor Flask importando app.py
    try:
        print("Importando e executando app.py...")
        import app # Importa app.py, que agora define a rota automaticamente
        
        # Executar o servidor usando os argumentos padrão
        app.app.run(host='0.0.0.0', port=port, debug=True)
        
    except ImportError:
        print("Erro: Não foi possível importar o módulo app.py.")
        print("Verifique se o arquivo app.py está no diretório correto.")
    except Exception as e:
        print(f"Erro ao executar app.py: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    iniciar_sistema() 