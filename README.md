# Sistema de Detecção de EPIs com YOLOv8

Sistema de detecção de Equipamentos de Proteção Individual (EPIs) utilizando YOLOv8 e Flask.

## Classes de Detecção

O sistema é capaz de detectar os seguintes EPIs:
- Pessoa
- Capacete de segurança
- Óculos de proteção
- Máscara
- Luvas de proteção
- Roupa de proteção
- Botas de segurança
- Abafador de ruído

## Requisitos

- Python 3.8+
- CUDA (opcional, para treinamento com GPU)
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/deteccao-epi.git
cd deteccao-epi
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Inicie o servidor Flask:
```bash
python novo_app.py
```

2. Acesse a interface web em `http://localhost:5000`

## Treinamento

Para treinar um novo modelo:

1. Prepare suas imagens e anotações no formato YOLO
2. Coloque as imagens na pasta `dataset_treinamento/images`
3. Coloque as anotações na pasta `dataset_treinamento/labels`
4. Acesse a interface de treinamento em `http://localhost:5000/treinamento`
5. Configure os parâmetros de treinamento e inicie

## Estrutura do Projeto

```
.
├── novo_app.py          # Aplicação principal Flask
├── treinamento.py       # Módulo de treinamento
├── dataset_config.yaml  # Configuração das classes
├── templates/          # Templates HTML
└── static/            # Arquivos estáticos
```

## Contribuição

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes. 