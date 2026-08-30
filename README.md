##🛩️  **Detector de Lixo com DJI Tello**

Projeto de visão computacional e controle de voo utilizando o drone DJI Tello, OpenCV e YOLOv8.

### 📌 **Requisitos**

- Python 3.10 ou superior
- DJI Tello
- Computador conectado à rede Wi-Fi do drone
- Git Bash ou PowerShell
- Câmera e conexão UDP disponíveis

### ⚙️ **Configuração**

Clone o projeto e entre na pasta:

```bash
git clone https://github.com/Jhonydev72/detector_de_lixo.git
cd detector_de_lixo
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual no Git Bash:

```bash
source .venv/Scripts/activate
```

No PowerShell, use:

```bash
Activate.ps1
```

Quando o ambiente estiver ativo, o terminal exibirá (.venv) no início da linha.

Instale as dependências:

```bash
pip install -r requirements.txt
```

### 🤖 **Modelo YOLO**

O arquivo `yolov8n.pt` deve estar na raiz do projeto. O script `teste_yolo_video.py` utiliza esse modelo para realizar detecção de objetos em tempo real.

### **Execução**

Conecte o computador à rede Wi-Fi do DJI Tello antes de executar os scripts.

**Teste básico de voo**

O script conecta ao drone, decola, gira 360 graus e pousa:

```bash
python main.py
```

**Teste de telemetria e voo**

```bash
python teste_voo.py
```

**Teste de vídeo**

```bash
python teste_video.py
```

Pressione `q` na janela do vídeo

Pressione `s` para salvar um frame como amostra_tello.jpg.

**Teste de voo com vídeo**

```bash
python teste_voo_video.py
```

**Teste integrado de voo e vídeo**

```bash
python teste_voo_video_definitivo.py
```

Pressione `q` para solicitar um pouso de emergência.

**Teste de vídeo via UDP**

```bash
python teste_udp_puro.py
```

Esse teste recebe diretamente o fluxo de vídeo UDP na porta `11111`.

**Detecção com YOLOv8**

```bash
python teste_yolo_video.py
```

Pressione `q` para encerrar a detecção.

**Coleta de imagens para dataset**

```bash
python coleta_dataset.py
```

As imagens serão salvas na pasta `dataset_lixo`.

**Controles durante a coleta:**

| Tecla | Ação |
|---|---|
| `T` | Decolar |
| `Y` | Pousar |
| `W` | Mover para frente |
| `S` | Mover para trás |
| `A` | Mover para a esquerda |
| `D` | Mover para a direita |
| `I` | Subir |
| `K` | Descer |
| `J` | Girar para a esquerda |
| `L` | Girar para a direita |
| `X` | Aumentar velocidade |
| `Z` | Diminuir velocidade |
| `C` | Capturar imagem |
| `Q` | Pousar e sair |

### ⚠️ **Segurança**

- Verifique se a bateria do drone está carregada.
- Execute os testes em uma área aberta e segura.
- Mantenha o drone longe de pessoas e obstáculos.
- Confirme que o computador está conectado à rede Wi-Fi do Tello.
- Tenha acesso ao pouso de emergência.
- Nunca execute testes de voo sem supervisão.
