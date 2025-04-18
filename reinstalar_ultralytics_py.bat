@echo off
echo Reinstalando Ultralytics e configurando ambiente (usando py)
echo ==========================================================

echo Verificando ambiente Python...
py --version
if %ERRORLEVEL% NEQ 0 (
    echo Erro: Python nao encontrado. Por favor, instale o Python 3.8 ou superior.
    pause
    exit /b 1
)

echo.
echo Instalando/Atualizando Ultralytics via pip...
py -m pip install ultralytics --upgrade

echo.
echo Configurando caminho padrao do dataset...
set SETTINGS_DIR=%APPDATA%\Ultralytics
if not exist "%SETTINGS_DIR%" mkdir "%SETTINGS_DIR%"

echo {^
    "datasets_dir": "%CD%"^
} > "%SETTINGS_DIR%\settings.json"

echo.
echo Verificando configuracao do dataset...
type dataset_config.yaml

echo.
echo Pronto! Ultralytics foi reinstalado e configurado.
echo O diretorio base de datasets foi definido como: %CD%
echo.
echo Agora, tente executar o treinamento novamente:
echo py novo_app.py
echo.
pause 