# LibrasVision

> Sistema de visão computacional para reconhecimento de sinais dinâmicos da Língua Brasileira de Sinais (Libras) em ambientes educacionais.

Trabalho de Conclusão de Curso — Ciência da Computação · Centro Universitário do Distrito Federal (UDF) · 2026

---

## Sobre o projeto

O **LibrasVision** propõe o desenvolvimento de um sistema capaz de reconhecer sinais dinâmicos da Libras e traduzi-los para texto em tempo real, com o objetivo de reduzir barreiras de comunicação entre estudantes surdos e a comunidade ouvinte no ambiente acadêmico.

A solução combina:
- **MediaPipe Holistic** — extração de 543 landmarks 3D por frame (33 pose + 468 face + 21 mão esq. + 21 mão dir.)
- **Rede LSTM** — modelagem temporal das sequências de landmarks para classificação dos sinais
- **Webcam convencional** — sem necessidade de hardware especializado

O repositório atual contém o **protótipo de demonstração (TCC 1)**, que valida o núcleo do pipeline: captura por webcam + extração de landmarks em tempo real + montagem da janela temporal de 30 frames. A implementação do modelo LSTM e a coleta do dataset ocorrem no TCC 2.

---

## Demonstração

O protótipo exibe ao vivo:

- Landmarks de mãos, rosto e pose desenhados sobre a imagem da webcam
- Contagem de landmarks visíveis por componente (meta: 543/543)
- Latência por frame em milissegundos (meta do projeto: < 100 ms)
- FPS em tempo real
- Barra de progresso da janela temporal de 30 frames que alimentará a rede LSTM

---

## Equipe

| Nome | GitHub |
|---|---|
| Augusto Cezar Araujo Saboia | — |
| Elizeu Barbosa Souza | — |
| Lorenzo Toledo | [@lorenzotoledo0](https://github.com/lorenzotoledo0) |
| Lucas Caitano Barreto | — |

**Orientadora:** Prof.ª Dr.ª Flavia Maria

---

## Requisitos

- Python **3.11** ou **3.12** (não compatível com 3.13+)
- Webcam USB ou integrada
- Windows 10+, Ubuntu 20.04+ ou macOS 11+

---

## Instalação

**1. Clone o repositório**

```bash
git clone https://github.com/lorenzotoledo0/LibrasVision.git
cd LibrasVision
```

**2. Crie o ambiente virtual com Python 3.12**

```bash
py -3.12 -m venv .venv
```

**3. Ative o ambiente virtual**

Windows:
```bash
.venv\Scripts\activate.bat
```

Linux/macOS:
```bash
source .venv/bin/activate
```

**4. Instale as dependências**

```bash
pip install -r requirements.txt
```

---

## Como rodar

```bash
python librasvision_demo.py
```

Opções disponíveis:

```bash
python librasvision_demo.py --camera 1       # câmera com índice 1 (se a 0 não abrir)
python librasvision_demo.py --complexity 0   # modo mais leve (se travar ou FPS baixo)
python librasvision_demo.py --width 1280 --height 720
```

---

## Controles

| Tecla | Ação |
|---|---|
| `q` / `ESC` | Sair |
| `h` | Mostra/oculta o painel (HUD) |
| `t` | Alterna malha facial completa |
| `m` | Alterna espelhamento da imagem |
| `f` | Tela cheia (útil no projetor) |
| `s` | Salva screenshot |
| `espaço` | Reinicia a janela de 30 frames |

---

## Estrutura do repositório

```
LibrasVision/
├── librasvision_demo.py   # protótipo de demonstração (TCC 1)
├── requirements.txt       # dependências Python
├── COMO_RODAR.txt         # instruções de uso em português
└── README.md
```

---

## Dependências

| Biblioteca | Versão | Finalidade |
|---|---|---|
| opencv-python | ≥ 4.5 | Captura e processamento de vídeo |
| mediapipe | 0.10.14 | Extração de landmarks em tempo real |
| numpy | ≥ 1.23 | Manipulação das sequências de landmarks |
| pillow | ≥ 9.0 | Renderização de texto no HUD (opcional) |

---

## Roadmap

- [x] Pipeline de captura por webcam (OpenCV)
- [x] Extração de 543 landmarks via MediaPipe Holistic
- [x] Janela temporal de 30 frames
- [x] Medição de latência por frame
- [ ] Coleta do dataset (biblioteca / prova / secretaria)
- [ ] Treinamento do modelo LSTM
- [ ] Avaliação: acurácia ≥ 85%, latência < 100 ms
- [ ] Interface de tradução em tempo real

---

## Referências

- KAMBLE, S. *SLRNet: A Real-Time LSTM-Based Sign Language Recognition System*. ArXiv, 2025.
- VARSHINI, T. S.; RUKMANI, P. *MP-GestLSTM: real time gesture detection using MediaPipe and LSTM*. Systems Science & Control Engineering, 2025.
- ANITHADEVI, N. et al. *MediaPipe-LSTM-enhanced framework for real-time dynamic sign language recognition*. Engineering Reports, 2025.
- COSTA, V. et al. *A lightweight I3D-based approach for real-time Brazilian sign language recognition*. WebMedia, 2025.

---

## Licença

Este projeto é de uso acadêmico. Todos os direitos reservados aos autores.
