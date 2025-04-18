================================================
  SISTEMA DE MONITORAMENTO DE EPIs - INSTRUÇÕES
================================================

REQUISITOS:
- Python instalado (comando 'py' disponível no Windows)
- Conexão com a internet (para baixar o modelo de IA)

PASSOS PARA INICIAR O SISTEMA:

1. INSTALAÇÃO DE DEPENDÊNCIAS
   - Execute o arquivo "instalar_dependencias.bat"
   - Aguarde a conclusão da instalação

2. INICIAR O SISTEMA
   - Execute o arquivo "iniciar.bat"
   - O sistema abrirá uma janela de terminal mostrando os endereços para acesso

3. ACESSAR A INTERFACE
   - No seu navegador, acesse um dos endereços exibidos:
     * Local: http://127.0.0.1:5000
     * Rede: http://SEU_IP:5000 (para acessar de outros dispositivos)

SOLUÇÃO DE PROBLEMAS:

1. Se ocorrerem erros ao carregar o modelo de IA:
   - Execute "iniciar_modo_simulacao.bat" para iniciar em modo de simulação
   - Este modo não utiliza IA real, mas permite testar a interface

2. Se ocorrerem erros de instalação:
   - Verifique se o Python está instalado corretamente
   - Tente executar os comandos manualmente no terminal:
     py -m pip install -r requirements.txt
     py app.py

3. Para acessar de outros dispositivos:
   - Certifique-se de que o firewall do Windows permite conexões na porta 5000
   - Todos os dispositivos devem estar na mesma rede Wi-Fi

================================================ 