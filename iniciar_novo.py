import os
import sys
import subprocess
import argparse

def verificar_dependencias():
    """Verifica se todas as dependências estão instaladas."""
    try:
        import flask
        import torch
        import ultralytics
        import PIL
        import numpy
        import requests
        print("✅ Todas as dependências estão instaladas.")
        return True
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        return False

def instalar_dependencias():
    """Instala as dependências necessárias."""
    print("📦 Instalando dependências...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "novo_requirements.txt"])
    print("✅ Dependências instaladas com sucesso!")

def criar_diretorios():
    """Cria os diretórios necessários para o aplicativo."""
    diretorios = ['uploads', 'resultados', 'models']
    for diretorio in diretorios:
        os.makedirs(diretorio, exist_ok=True)
        print(f"✅ Diretório '{diretorio}' verificado/criado.")

def main():
    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Inicia o detector de objetos YOLO.')
    parser.add_argument('--porta', type=int, default=5000, help='Porta para o servidor (padrão: 5000)')
    parser.add_argument('--host', default='0.0.0.0', help='Host para o servidor (padrão: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Ativar modo de debug')
    parser.add_argument('--instalar', action='store_true', help='Instalar dependências')
    
    args = parser.parse_args()
    
    print("\n🚀 Inicializando detector de objetos com YOLO...\n")
    
    # Verificar e instalar dependências se necessário
    if args.instalar or not verificar_dependencias():
        instalar_dependencias()
        
    # Criar diretórios necessários
    criar_diretorios()
    
    # Verificar arquivos críticos
    if not os.path.exists('novo_app.py'):
        print("❌ Erro: arquivo 'novo_app.py' não encontrado!")
        return
    
    if not os.path.exists('templates/index.html'):
        print("❌ Erro: arquivo 'templates/index.html' não encontrado!")
        return
    
    # Configurar variáveis de ambiente
    os.environ['PORT'] = str(args.porta)
    os.environ['FLASK_APP'] = 'novo_app.py'
    
    if args.debug:
        os.environ['FLASK_ENV'] = 'development'
        os.environ['FLASK_DEBUG'] = '1'
    
    print(f"\n🌐 Iniciando servidor na porta {args.porta}...")
    
    # Iniciar o aplicativo Flask
    try:
        subprocess.run([sys.executable, 'novo_app.py'])
    except KeyboardInterrupt:
        print("\n\n🛑 Servidor encerrado pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro ao iniciar o servidor: {e}")

if __name__ == "__main__":
    main() 