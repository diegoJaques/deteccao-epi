@echo off
echo.
echo ========================================
echo   Inicializando Detector YOLO - Novo
echo ========================================
echo.

:: Verificar se Python está instalado
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python não encontrado. Por favor, instale o Python 3.8 ou superior.
    pause
    exit /b
)

:: Iniciar o aplicativo
echo Iniciando o Detector YOLO...
echo.
python iniciar_novo.py --debug

:: Se o script terminar com erro, não fechar a janela
if %errorlevel% neq 0 (
    echo.
    echo [ERRO] O aplicativo encerrou com erros.
    pause
) 