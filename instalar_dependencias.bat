@echo off
echo ================================================
echo     INSTALAÇÃO DE DEPENDÊNCIAS
echo ================================================
echo.

echo Instalando dependências necessárias...
py -m pip install --upgrade pip
py -m pip install opencv-python==4.8.0.76
py -m pip install numpy==1.24.3
py -m pip install flask==2.3.3
py -m pip install pillow==10.0.0
py -m pip install python-dotenv==1.0.0
py -m pip install torch==2.0.1
py -m pip install transformers==4.31.0

echo.
echo Instalação concluída!
echo.
pause 