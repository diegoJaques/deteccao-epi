@echo off
echo Configurando Ultralytics e iniciando aplicacao
echo ==========================================

echo.
echo Passo 1: Configurando o Ultralytics...
py configurar_ultralytics.py

echo.
echo Passo 2: Verificando o ambiente...
if not exist "dataset_treinamento" (
    echo Pasta dataset_treinamento nao encontrada, criando...
    mkdir dataset_treinamento
)
if not exist "dataset_treinamento\images" mkdir dataset_treinamento\images
if not exist "dataset_treinamento\labels" mkdir dataset_treinamento\labels

echo.
echo Passo 3: Iniciando a aplicacao...
py novo_app.py

pause 