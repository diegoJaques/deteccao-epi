@echo off
echo Iniciando aplicacao usando o comando 'py'...
echo =========================================

echo.
echo Verificando pasta dataset_treinamento...
if not exist "dataset_treinamento" (
    echo Criando pasta dataset_treinamento...
    mkdir dataset_treinamento
)

if not exist "dataset_treinamento\images" (
    echo Criando pasta dataset_treinamento\images...
    mkdir dataset_treinamento\images
)

if not exist "dataset_treinamento\labels" (
    echo Criando pasta dataset_treinamento\labels...
    mkdir dataset_treinamento\labels
)

echo.
echo Verificando arquivo de configuracao...
echo Conteudo de dataset_config.yaml:
type dataset_config.yaml

echo.
echo Iniciando a aplicacao...
py novo_app.py

pause 