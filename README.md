# 🛰️ VANT Edge AI: Sistema Autônomo de Inspeção Ambiental e Detecção de Resíduos com DJI Tello e YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DJI Tello](https://img.shields.io/badge/Hardware-DJI%20Tello-00A4E4.svg)](https://www.ryzerobotics.com/tello)
[![PostGIS](https://img.shields.io/badge/Database-PostgreSQL%20%2F%20PostGIS-336791.svg?logo=postgresql&logoColor=white)](https://postgis.net/)
[![Licença](https://img.shields.io/badge/Status-TCC%20%2F%20Engenharia-brightgreen.svg)]()

> **Projeto de Conclusão de Curso (TCC)** focado no desenvolvimento de uma arquitetura de computação de borda (*Edge Computing*) para veículos aéreos não tripulados (VANTs), realizando voo controlado, geolocalização estimada por odometria inercial e detecção de resíduos sólidos em tempo real utilizando redes neurais convolucionais (YOLOv8).

---

## 📋 Sumário

- [Visão Geral e Arquitetura](#-visão-geral-e-arquitetura)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Treinamento da Inteligência Artificial (YOLOv8)](#-treinamento-da-inteligência-artificial-yolov8)
  - [1. Coleta e Construção do Dataset](#1-coleta-e-construção-do-dataset)
  - [2. Configuração das Classes (data.yaml)](#2-configuração-das-classes-datayaml)
  - [3. Arquitetura da Rede e Transfer Learning](#3-arquitetura-da-rede-e-transfer-learning)
  - [4. Execução do Treinamento na GPU](#4-execução-do-treinamento-na-gpu)
  - [5. Métricas de Desempenho e Resultados Reais](#5-métricas-de-desempenho-e-resultados-reais)
- [Suíte de Testes e Validação Incremental](#-suíte-de-testes-e-validação-incremental)
  - [Teste 1: Hardware e Telemetria Básica (`teste_voo.py`)](#teste-1-hardware-e-telemetria-básica-teste_voopy)
  - [Teste 2: Ingestão de Vídeo UDP (`teste_video.py`)](#teste-2-ingestão-de-vídeo-udp-teste_videopy)
  - [Teste 3: Voo Concorrente e Multithreading (`teste_voo_video.py`)](#teste-3-voo-concorrente-e-multithreading-teste_voo_videopy)
  - [Teste 4: Arquitetura Definitiva Tolerante a Falhas (`teste_voo_video_definitivo.py`)](#teste-4-arquitetura-definitiva-tolerante-a-falhas-teste_voo_video_definitivopy)
  - [Teste 5: Inferência de IA em Tempo Real (`teste_yolo_video.py`)](#teste-5-inferência-de-ia-em-tempo-real-teste_yolo_videopy)
  - [Sistema Integrado Final (`voo_manual_ia.py`)](#sistema-integrado-final-voo_manual_iapy)
- [Georreferenciamento por Odometria e Banco PostGIS](#-georreferenciamento-por-odometria-e-banco-postgis)
- [Guia de Instalação e Execução](#-guia-de-instalação-e-execução)
- [Mapeamento de Teclas (Controles do Operador)](#-mapeamento-de-teclas-controles-do-operador)

---

## 🏛️ Visão Geral e Arquitetura

O sistema implementa o paradigma de **Edge Computing**. Como o microprocessador embarcado do drone DJI Tello possui restrições severas de peso, memória e capacidade computacional, todo o processamento pesado de visão computacional e controle é descarregado em tempo real para uma **Estação Base Local (Edge Base Station)** munida de GPU dedicada:

```
      +------------------------+
      |    DJI Tello (VANT)    |
      | - Câmera HD 720p       |
      | - Barômetro e ToF      |
      | - IMU (Velocidades)    |
      +-----------+------------+
                  |
                  |  Wi-Fi 802.11n (UDP)
                  |  - Porta 8889: Comandos / Telemetria
                  |  - Porta 11111: Stream de Vídeo H.264
                  v
      +---------------------------------------------------+
      |        Estação Base Edge (PC Local / GPU)        |
      |                                                   |
      |  [ Thread de Voo & Telemetria ]                   |
      |   -> djitellopy: Takeoff, Land, RC Control, ToF   |
      |   -> Odometria Inercial: Dead Reckoning (Lat/Lon) |
      |                                                   |
      |  [ Thread de Ingestão de Vídeo ]                  |
      |   -> OpenCV + FFMPEG (Zero Buffer, Low Delay)     |
      |                                                   |
      |  [ Pipeline de Visão Computacional ]              |
      |   -> YOLOv8 Nano (Inferência acelerada por CUDA)  |
      |   -> HUD em tempo real com caixas delimitadoras   |
      +---------------------------+-----------------------+
                                  |
                                  v
      +---------------------------------------------------+
      |      Persistência Espacial (PostgreSQL/PostGIS)   |
      |  - Tabela Voo, Mapeamento, Deteccao, Recorrencia  |
      +---------------------------------------------------+
```

---

## 📁 Estrutura do Repositório

```text
├── data/
│   └── tabela_postegis.sql            # Script DDL com extensão espacial PostGIS e tabelas relacionais
├── dataset_multiclasse/               # Dataset anotado multiclasse (YOLO format)
│   ├── data.yaml                      # Configuração das classes e caminhos de treino/validação
│   ├── images/
│   │   ├── train/                     # 292 imagens de treinamento
│   │   └── val/                       # 73 imagens de validação
│   └── labels/
│       ├── train/                     # Anotações .txt das imagens de treino
│       └── val/                       # Anotações .txt das imagens de validação
├── dataset_lixo_yolo/                 # Dataset histórico monoclasse (Saco de Lixo)
├── runs/                              # Saídas de treinamento do YOLOv8 multiclasse
│   └── detect/
│       └── modelos_tcc/
│           └── yolo_multiclasse_v1-5/ # Pesos (best.pt/last.pt), gráficos e matriz de confusão
├── runs_antiga/                       # Histórico dos primeiros treinamentos do modelo
├── coleta_dataset.py                  # Script de voo manual com captura de frames para rotulagem
├── treinar_modelo.py                  # Script base para treino do modelo na GPU
├── treinar_modelo_atualizado.py       # Script com isolamento de diretórios para treino multiclasse
├── teste_voo.py                       # Sprint 1: Conexão, decolagem, telemetria ToF e pouso seguro
├── teste_video.py                     # Sprint 2: Captura e decodificação de pacotes de vídeo UDP
├── teste_voo_video.py                 # Sprint 3: Multithreading voo + vídeo sem travamento
├── teste_voo_video_definitivo.py      # Sprint 3 Definitivo: Otimização de buffer e HUD resiliente
├── teste_yolo_video.py                # Sprint 4: Inferência YOLOv8 em tempo real no feed de vídeo
├── voo_manual_ia.py                   # Sprint 5: Sistema integrado com controle, IA e Odometria/GPS
├── requirements.txt                   # Dependências do ambiente Python (PyTorch CUDA, Ultralytics, etc.)
└── .gitignore                         # Regras de exclusão de binários, ambientes e dados pesados
```

---

## 🧠 Treinamento da Inteligência Artificial (YOLOv8)

O componente central do sistema de visão computacional é a detecção e classificação em tempo real de descartes irregulares de lixo. A rede foi concebida para operar com **alta taxa de quadros por segundo (FPS)** e **baixa latência**, fundamentais para a estabilidade do voo e resposta do operador.

### 1. Coleta e Construção do Dataset

Para garantir a generalização do modelo à perspectiva aérea do drone (ângulos cenitais e plongée, iluminação variável e texturas de solo), o próprio sistema conta com o módulo [`coleta_dataset.py`](coleta_dataset.py):

- **Captura em Voo:** O operador pilota o drone pelo teclado e, ao pressionar a tecla `C`, o frame de alta definição é capturado e salvo na pasta com nomenclatura baseada em timestamp (`saco_lixo_<timestamp>.jpg`).
- **Anotação:** As imagens foram rotuladas no padrão padrão YOLO, gerando arquivos `.txt` correspondentes com coordenadas normalizadas `[class_id, x_center, y_center, width, height]` no intervalo $[0, 1]$.
- **Divisão do Dataset Multiclasse:**
  - **Treinamento (`train`):** 292 imagens (80%)
  - **Validação (`val`):** 73 imagens (20%)
  - **Total de Amostras:** 365 imagens reais

### 2. Configuração das Classes (`data.yaml`)

O arquivo [`dataset_multiclasse/data.yaml`](dataset_multiclasse/data.yaml) define o mapeamento das 4 categorias de resíduos investigadas:

```yaml
train: C:/Users/JoãoF/Desktop/drone/dataset_multiclasse/images/train
val: C:/Users/JoãoF/Desktop/drone/dataset_multiclasse/images/val

nc: 4
names: ['Saco de lixo', 'Garrafa PET', 'Latinha', 'Pneu']
```

> **Atenção:** Em ambientes Windows com caminhos que contenham acentuações ou caracteres especiais, declaram-se caminhos absolutos com barras normais (`/`) sem a variável de prefixo `path:` para evitar falhas de concatenação interna do motor C++ do Ultralytics.

### 3. Arquitetura da Rede e Transfer Learning

- **Modelo Base:** `YOLOv8n` (*YOLOv8 Nano*), contendo cerca de 3,2 milhões de parâmetros.
- **Justificativa da Escolha:** A variante Nano apresenta o menor consumo de memória de vídeo (VRAM) e o tempo de inferência mais veloz da família YOLOv8 (< 6 ms em GPU dedicada), viabilizando taxas superiores a 30 FPS simultaneamente ao streaming UDP e à renderização da interface OpenCV.
- **Transferência de Aprendizado:** O treinamento parte dos pesos pré-treinados no *MS COCO* (`yolov8n.pt`), aproveitando extratores de características de baixo e médio nível já consolidados e refinando as camadas de detecção (*head*) para os resíduos de interesse.

### 4. Execução do Treinamento na GPU

O processo de treinamento é orquestrado pelo script [`treinar_modelo_atualizado.py`](treinar_modelo_atualizado.py):

```python
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

def iniciar_treinamento_gpu():
    logging.info("Carregando modelo base YOLOv8 Nano...")
    modelo = YOLO('yolov8n.pt') 

    logging.info("Iniciando treinamento multiclasse acelerado na RTX 2050 (GPU 0)...")
    resultados = modelo.train(
        data='C:/Users/JoãoF/Desktop/drone/dataset_multiclasse/data.yaml', 
        epochs=50,
        imgsz=640, 
        plots=True,
        device=0,                   # Força uso exclusivo da GPU NVIDIA (CUDA)
        project='modelos_tcc',      # Diretório organizado de saídas
        name='yolo_multiclasse_v1'  # Identificador da rodada
    )
    logging.info("Treinamento finalizado! Pesos salvos em 'modelos_tcc/yolo_multiclasse_v1/weights/best.pt'")

if __name__ == "__main__":
    iniciar_treinamento_gpu()
```

#### Hiperparâmetros Chave:
| Parâmetro | Valor | Descrição |
| :--- | :--- | :--- |
| **Épocas (`epochs`)** | `50` | Ciclos completos de passagem pelos dados de treino |
| **Resolução (`imgsz`)** | `640x640` | Dimensão das imagens durante o treinamento |
| **Dispositivo (`device`)** | `0` | GPU NVIDIA GeForce RTX 2050 (via CUDA) |
| **Otimizador** | `Auto (SGD/AdamW)` | Ajuste automático da taxa de aprendizado e decaimento |
| **Augmentation** | Ativado nativo | Mosaico, rotação leve, variação de escala e brilho |

### 5. Métricas de Desempenho e Resultados Reais

Durante as 50 épocas de treinamento, o comportamento de convergência foi registrado nos relatórios de validação em `runs/detect/modelos_tcc/`:

#### Evolução das Rodadas de Treinamento:

| Modelo | Dataset | Classes | Precisão ($P$) | Recall ($R$) | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Monoclasse (Baseline)** | `dataset_lixo_yolo` | 1 (*Saco de lixo*) | **98.7%** | **94.6%** | **99.2%** | **87.0%** |
| **Multiclasse (Final)** | `dataset_multiclasse` | 4 (*Saco, Garrafa, Lata, Pneu*) | **77.2%** | **82.7%** | **81.3%** | **58.3%** |

#### Interpretação das Métricas Técnicas:
- **Precisão ($P = \frac{TP}{TP + FP}$):** $77.2\%$ no modelo multiclasse indica que a grande maioria das detecções apontadas pelo drone é de fato um resíduo real, minimizando falsos positivos durante a varredura.
- **Revocação / Recall ($R = \frac{TP}{TP + FN}$):** $82.7\%$ demonstra alta capacidade de localização dos objetos mesmo em cenários com oclusão parcial ou vegetação.
- **mAP@0.5:** $81.3\%$ de precisão média sob Intersection over Union (IoU) de $0.50$, assegurando enquadramento preciso do resíduo.
- **Localização dos Pesos:**
  - `best.pt`: Pesos da época que alcançou a melhor métrica no conjunto de validação (utilizado na inspeção real).
  - `last.pt`: Pesos da última época de treinamento (útil para retomadas).

---

## 🧪 Suíte de Testes e Validação Incremental

O desenvolvimento do sistema de voo e processamento foi estruturado de forma estritamente modular por meio de sprints de validação de software e hardware, garantindo a integridade física do VANT e a estabilidade da comunicação UDP:

```
[ Sprint 1: Hardware/Voo ] ──> [ Sprint 2: Vídeo UDP ] ──> [ Sprint 3: Multithreading ]
                                                                       │
[ Sprint 5: Sistema Completo ] <── [ Sprint 4: Inferência IA ] <───────┘
```

---

### Teste 1: Hardware e Telemetria Básica ([`teste_voo.py`](teste_voo.py))
- **Objetivo:** Estabelecer o handshake com o firmware do drone via sockets UDP (porta 8889), validar as leituras dos sensores e homologar o ciclo de vida essencial de voo (decolagem $\rightarrow$ voo pairado $\rightarrow$ pouso).
- **Mecanismos de Segurança:**
  - Checagem prévia da bateria: aborta caso a carga seja inferior a $20\%$.
  - Bloco `try...finally`: garante que o comando de pouso (`land()`) e encerramento de conexão (`end()`) sejam emitidos impreterivelmente, mesmo em caso de exceções inesperadas em tempo de execução.
- **Leituras Homologadas:** Sensor de tempo de voo (ToF em centímetros), percentual de bateria e temperatura interna dos circuitos.

---

### Teste 2: Ingestão de Vídeo UDP ([`teste_video.py`](teste_video.py))
- **Objetivo:** Capturar e decodificar os pacotes de vídeo H.264 transmitidos na porta UDP `11111` sem o uso de dependências externas bloqueantes.
- **Tratamento Específico para Windows e FFMPEG:**
  - Utilização da sintaxe `"udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=50000000"` com o prefixo `@`, garantindo *bind* local de socket sem bloqueio no subsistema de rede do Windows.
  - Implementação de timeout por contagem de quadros vazios com alerta contra conflitos de interface de rede (por exemplo, placas Ethernet ativas competindo com a rede Wi-Fi do Tello).
  - Funcionalidade de captura pontual de frame (tecla `S`) para salvar amostras (`amostra_tello.jpg`).

---

### Teste 3: Voo Concorrente e Multithreading ([`teste_voo_video.py`](teste_voo_video.py))
- **Objetivo:** Resolver o problema do travamento de renderização da interface gráfica do OpenCV durante a execução de rotinas bloqueantes de voo (`time.sleep` e comandos de movimentação síncronos).
- **Solução Implementada:**
  - **Thread Secundária (`threading.Thread` em background):** Executa os comandos sequenciais de decolagem, espera e pouso.
  - **Thread Principal:** Executa ininterruptamente a captura de frames (`get_frame_read()`), redimensionamento e exibição na janela OpenCV (~30 FPS).
  - Adição de HUD com estado do voo e tecla de emergência `Q` para pouso manual instantâneo.

---

### Teste 4: Arquitetura Definitiva Tolerante a Falhas ([`teste_voo_video_definitivo.py`](teste_voo_video_definitivo.py))
- **Objetivo:** Eliminar qualquer latência acumulada no buffer de imagem (*frame lag*) e blindar o sistema contra oscilações de sinal Wi-Fi.
- **Otimizações do Kernel de Captura:**
  - Configuração de flag de ambiente de baixo nível antes da carga do OpenCV:
    ```python
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"
    ```
  - Configuração `get_frame_read(with_queue=False)`: desativa a fila de frames acumulados, processando exclusivamente o quadro mais recente recebido pela placa de rede.
  - Indicador de resfriamento ativo no HUD: monitora a ventilação gerada pelo fluxo das hélices em voo para evitar superaquecimento da placa do Tello quando parado em bancada.

---

### Teste 5: Inferência de IA em Tempo Real ([`teste_yolo_video.py`](teste_yolo_video.py))
- **Objetivo:** Acoplar a rede neural YOLOv8 ao pipeline de vídeo direto da câmera do drone, mensurando a taxa de quadros e o comportamento das caixas delimitadoras.
- **Pipeline:**
  1. Captura do frame em resolução nativa;
  2. Redimensionamento rápido para $720\times 480$ pixels;
  3. Inferência através do modelo `YOLO('yolov8n.pt')` com `verbose=False`;
  4. Sobreposição visual das previsões (`resultados[0].plot()`) e exibição na estação de controle.

---

### Sistema Integrado Final ([`voo_manual_ia.py`](voo_manual_ia.py))
- **Objetivo:** O módulo culminante do projeto. Unifica a pilotagem bidirecional em seis eixos, a inferência do modelo multiclasse treinado (`best.pt`), odometria inercial e telemetria completa em tela.
- **Principais Funcionalidades:**
  - Carregamento automático dos pesos otimizados com filtro de confiança (`conf=0.60`), suprimindo ruídos e detecções espúrias.
  - Painel HUD dinâmico informando status da IA, nível de bateria, velocidade atual de cruzeiro e coordenadas geográficas simuladas.
  - Controle manual com resposta em tempo real a comandos combinados de translação e rotação.

---

## 🗺️ Georreferenciamento por Odometria e Banco PostGIS

### 1. Cálculo de Odometria Inercial (*Dead Reckoning*)

Como o DJI Tello não possui receptor GNSS/GPS embarcado de fábrica, o sistema implementa estimativa de posição geográfica a partir da integração temporal das velocidades inerciais reportadas pela unidade de medição inercial (IMU) do drone:

$$\Delta x = v_x \cdot \Delta t, \quad \Delta y = v_y \cdot \Delta t$$

Onde $v_x$ e $v_y$ são lidos via `tello.get_speed_x()` e `tello.get_speed_y()`, e $\Delta t$ é a variação temporal entre os quadros.

A translação métrica acumulada $(x, y)$ em centímetros é convertida para coordenadas geográficas a partir de um ponto de referência inicial:

$$\text{Latitude}_{\text{atual}} = \text{Lat}_{\text{inicial}} + \left(\frac{\text{pos}_x}{100.0} \times 0.000009\right)$$
$$\text{Longitude}_{\text{atual}} = \text{Lon}_{\text{inicial}} + \left(\frac{\text{pos}_y}{100.0} \times 0.000009\right)$$

*(Constante de conversão: $\approx 1\text{ metro} \approx 0.000009^\circ$ de arco geográfico).*

### 2. Modelagem Espacial PostGIS ([`data/tabela_postegis.sql`](data/tabela_postegis.sql))

As infrações e eventos de mapeamento são modelados relacionalmente com suporte à extensão espacial do PostgreSQL:

```sql
-- Habilita o motor espacial
CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Tabela Voo: Registra a sessão de inspeção ou mapeamento
CREATE TABLE Voo (
    id_voo SERIAL PRIMARY KEY,
    data_voo TIMESTAMP NOT NULL,
    tipo_voo VARCHAR(20) NOT NULL CHECK (tipo_voo IN ('INSPECAO', 'MAPEAMENTO_2D'))
);

-- 2. Tabela Deteccao: Armazena as ocorrências de resíduos detectadas pelo YOLOv8
CREATE TABLE Deteccao (
    id_deteccao SERIAL PRIMARY KEY,
    id_voo INT NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    confianca_ia DECIMAL(3, 2) NOT NULL CHECK (confianca_ia >= 0.00 AND confianca_ia <= 1.00),
    caminho_imagem VARCHAR(255) NOT NULL,
    CONSTRAINT fk_voo_deteccao FOREIGN KEY (id_voo) REFERENCES Voo(id_voo) ON DELETE RESTRICT
);

-- 3. Tabela Recorrencia: Inteligência espacial para cruzamento de descartes reincidentes
CREATE TABLE Recorrencia (
    id_recorrencia SERIAL PRIMARY KEY,
    id_deteccao_hoje INT NOT NULL,
    id_deteccao_passado INT NOT NULL,
    CONSTRAINT fk_deteccao_atual FOREIGN KEY (id_deteccao_hoje) REFERENCES Deteccao(id_deteccao),
    CONSTRAINT fk_deteccao_passado FOREIGN KEY (id_deteccao_passado) REFERENCES Deteccao(id_deteccao),
    CONSTRAINT chk_deteccoes_diferentes CHECK (id_deteccao_hoje != id_deteccao_passado)
);
```

---

## 🚀 Guia de Instalação e Execução

### Pré-requisitos
1. **Sistema Operacional:** Windows 10/11 ou Linux.
2. **Placa de Vídeo (Recomendado):** NVIDIA GeForce com suporte a CUDA (ex: RTX 2050, 3060, etc.).
3. **Python:** Versão 3.10 ou 3.11 de 64 bits.
4. **Hardware:** Drone DJI Tello e bateria com carga superior a $50\%$.

### 1. Clonar o Repositório e Criar Ambiente Virtual
```powershell
# Criação do ambiente virtual
python -m venv venv

# Ativação do ambiente virtual no Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 2. Instalação das Dependências
Instale as bibliotecas requeridas presentes no arquivo `requirements.txt`:
```powershell
pip install -r requirements.txt
```

> **Verificação de Aceleração CUDA:**
> Certifique-se de que o PyTorch identificou a sua GPU executando no terminal:
> ```powershell
> python -c "import torch; print('CUDA disponível:', torch.cuda.is_available(), '| Dispositivo:', torch.cuda.get_device_name(0))"
> ```

### 3. Conexão com o Drone
1. Ligue o DJI Tello e aguarde o LED frontal piscar em amarelo rápido.
2. Conecte o Wi-Fi do seu computador à rede aberta do drone (`TELLO-XXXXXX`).
3. *(Opcional no Windows)*: Desative temporariamente a placa de rede Ethernet cabeada se houver interferência na rota de roteamento local de pacotes UDP.

### 4. Executando os Scripts

- **Para rodar o teste básico de decolagem e telemetria:**
  ```powershell
  python teste_voo.py
  ```

- **Para testar a transmissão do vídeo:**
  ```powershell
  python teste_voo_video_definitivo.py
  ```

- **Para coletar fotos para o dataset durante o voo:**
  ```powershell
  python coleta_dataset.py
  ```

- **Para treinar o modelo multiclasse na GPU:**
  ```powershell
  python treinar_modelo_atualizado.py
  ```

- **Para executar a missão completa de inspeção com IA e Odometria:**
  ```powershell
  python voo_manual_ia.py
  ```

---

## 🎮 Mapeamento de Teclas (Controles do Operador)

Ao executar os scripts de pilotagem ([`coleta_dataset.py`](coleta_dataset.py) e [`voo_manual_ia.py`](voo_manual_ia.py)), mantenha o foco na janela gráfica do OpenCV:

| Tecla | Função de Controle |
| :---: | :--- |
| `T` | **Decolar (*Takeoff*)** |
| `Y` | **Pousar (*Land*)** |
| `W` / `S` | Mover para **Frente** / **Trás** |
| `A` / `D` | Mover para a **Esquerda** / **Direita** |
| `I` / `K` | Mover para **Cima** / **Baixo** (Altitude) |
| `J` / `L` | **Girar no próprio eixo (*Yaw*)** Anti-horário / Horário |
| `X` / `Z` | **Aumentar** / **Diminuir** sensibilidade de velocidade ($\pm 10$) |
| `C` | **Capturar Frame** (*apenas em `coleta_dataset.py`*) |
| `Q` | **Pouso de Emergência e Encerramento Imediato** |

---

## 👥 Autoria e Agradecimentos

Desenvolvido no contexto de Trabalho de Conclusão de Curso (TCC), explorando a aplicação de Veículos Aéreos Não Tripulados e Visão Computacional para Cidades Inteligentes e Monitoramento Ambiental Urbano.
