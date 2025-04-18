@echo off
echo ================================================
echo     SISTEMA DE MONITORAMENTO DE EPIs
echo ================================================
echo.
echo Iniciando o sistema...
echo.

rem Verificar dependências
py -m pip install -r requirements.txt

rem Iniciar o servidor usando o script Python principal
rem Isso garante que as verificações e a inicialização correta ocorram
py iniciar.py

echo.
echo Se ocorrer algum erro, tente executar os comandos:
echo   py -m pip install -r requirements.txt
echo   py iniciar.py
echo.
pause 