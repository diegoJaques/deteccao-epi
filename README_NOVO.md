# Detector de Objetos com YOLO

Este projeto é uma aplicação web simples para detecção de objetos em imagens utilizando modelos YOLO da [Ultralytics](https://github.com/ultralytics/ultralytics).

## Funcionalidades

- Upload de imagens por arrastar-e-soltar ou seleção de arquivo
- Detecção de objetos com modelos YOLO
- Visualização dos resultados com bounding boxes
- Gerenciamento de múltiplos modelos YOLO
- Ajuste de nível de confiança para detecções
- Informações do sistema (hardware, GPU, etc.)

## Requisitos

- Python 3.8 ou superior
- Biblioteca Ultralytics (YOLO)
- Flask para o servidor web
- OpenCV, NumPy e Pillow para processamento de imagens
- Conexão com a internet (para download inicial de modelos)

## Instalação

1. Clone este repositório ou baixe os arquivos
2. Instale as dependências:
   ```
   pip install -r novo_requirements.txt
   ```
   
   Ou inicie o script com a flag `--instalar`:
   ```
   python iniciar_novo.py --instalar
   ```

## Uso

### Iniciar o aplicativo

No Windows:
```
iniciar_novo.bat
```

Ou usando Python diretamente:
```
python iniciar_novo.py
```

### Flags de linha de comando

Você pode personalizar a inicialização com as seguintes flags:

- `--porta NUMERO`: Define a porta do servidor (padrão: 5000)
- `--host ENDERECO`: Define o host do servidor (padrão: 0.0.0.0)
- `--debug`: Ativa o modo de debug
- `--instalar`: Instala as dependências necessárias

Exemplo:
```
python iniciar_novo.py --porta 8000 --debug
```

### Interface Web

Após iniciar o servidor, acesse a interface web em:
```
http://localhost:5000
```

## Detecção de Objetos

1. Arraste e solte uma imagem na área designada ou clique em "Selecionar Imagem"
2. Ajuste o nível de confiança desejado (padrão: 0.25)
3. Clique em "Detectar Objetos"
4. Após o processamento, os resultados serão exibidos com as detecções encontradas

## Gerenciamento de Modelos

A aplicação suporta diversos modelos YOLO:

- **YOLOv8n**: modelo mais rápido, menor precisão
- **YOLOv8s**: equilíbrio entre velocidade e precisão
- **YOLOv8m**: precisão média, velocidade moderada
- **YOLOv8l**: alta precisão, menor velocidade
- **YOLOv8x**: maior precisão, velocidade mais lenta

Você pode adicionar seus próprios modelos YOLO treinados colocando os arquivos `.pt` na pasta `models/`.

## Estrutura de Diretórios

- `uploads/`: Armazena as imagens enviadas para detecção
- `resultados/`: Armazena as imagens com as detecções desenhadas
- `models/`: Armazena os modelos YOLO (arquivos .pt)
- `templates/`: Contém os arquivos HTML da interface web

## API

A aplicação também fornece uma API simples:

- `GET /`: Interface web principal
- `GET /modelos`: Lista os modelos disponíveis
- `POST /definir_modelo`: Define o modelo ativo
- `POST /detectar`: Realiza detecção em uma imagem
- `GET /info`: Retorna informações do sistema

---

Baseado na biblioteca [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). 