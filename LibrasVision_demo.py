#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LibrasVision - Protótipo de demonstração (TCC 1)
=================================================
Captura por webcam + extração de landmarks via MediaPipe Holistic em tempo real.

Este protótipo demonstra o NÚCLEO da metodologia do LibrasVision:
  - Captura de vídeo por webcam convencional (OpenCV);
  - Extração dos 543 landmarks 3D por frame via MediaPipe Holistic
    (33 pose + 468 face + 21 mão esquerda + 21 mão direita);
  - Montagem da janela temporal de 30 frames que alimentaria a rede LSTM (TCC 2);
  - Medição da latência por frame (meta do projeto: < 100 ms).

IMPORTANTE: a CLASSIFICAÇÃO do sinal NÃO é feita aqui. O modelo LSTM e o
dataset são desenvolvidos no TCC 2. O painel "Tradução" indica exatamente
onde esse módulo se conecta ao pipeline - este protótipo prova que a etapa
de captura e extração de características funciona em hardware convencional.

Controles (com a janela em foco):
  q / ESC  -> sair
  h        -> mostra / oculta o painel (HUD)
  t        -> alterna malha facial completa (tesselation)
  m        -> alterna espelhamento da imagem
  f        -> alterna tela cheia (útil no projetor)
  s        -> salva um print (screenshot) para os slides
  espaço   -> reinicia a janela de 30 frames

Uso:
  python librasvision_demo.py
  python librasvision_demo.py --camera 1 --width 1280 --height 720
  python librasvision_demo.py --complexity 0      (mais leve, se travar)

Dependências:
  pip install "opencv-python>=4.5" "mediapipe>=0.10.9,<0.11" "numpy>=1.23"
  (opcional, para textos com acento bonitos no HUD: pip install pillow)
"""

import argparse
import os
import platform
import sys
import time
import unicodedata
from collections import deque

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("[erro] OpenCV nao encontrado. Instale com: pip install opencv-python")

try:
    import mediapipe as mp
except ImportError:
    sys.exit('[erro] MediaPipe nao encontrado. Instale com: pip install "mediapipe>=0.10.9,<0.11"')

# Pillow é opcional: melhora o texto do HUD (acentos). Sem ele, cai para ASCII.
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# --------------------------------------------------------------------------- #
# Constantes do projeto
# --------------------------------------------------------------------------- #
N_POSE = 33
N_FACE = 468
N_HAND = 21
N_TOTAL = N_POSE + N_FACE + 2 * N_HAND          # 543 landmarks
N_FEATURES = N_TOTAL * 3                         # 1629 valores (x, y, z) por frame
SEQ_LEN = 30                                     # janela temporal (30 frames @ 30 FPS)

# Paleta (BGR)
COR_PAINEL = (30, 28, 28)
COR_ACENTO = (45, 45, 225)      # vermelho (branding LibrasVision)
COR_OK = (95, 205, 95)
COR_OFF = (95, 95, 95)
COR_TXT = (240, 240, 240)
COR_TXT2 = (180, 180, 180)
COR_BARRA_BG = (60, 60, 60)

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# --------------------------------------------------------------------------- #
# Renderização de texto (PIL com fallback ASCII via OpenCV)
# --------------------------------------------------------------------------- #
def _find_font():
    candidatos = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidatos:
        if os.path.exists(p):
            return p
    return None


def _strip_accents(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


class TextHUD:
    """Acumula textos e os desenha de uma vez por frame (1 conversao PIL)."""

    def __init__(self):
        self.itens = []  # (texto, x, y, tamanho, cor_bgr, negrito)
        self._font_path = _find_font() if _PIL_OK else None
        self._usar_pil = _PIL_OK and self._font_path is not None
        self._cache = {}

    def add(self, texto, x, y, tam=18, cor=COR_TXT, negrito=False):
        self.itens.append((texto, int(x), int(y), int(tam), cor, negrito))

    def _font(self, tam):
        if tam not in self._cache:
            self._cache[tam] = ImageFont.truetype(self._font_path, tam)
        return self._cache[tam]

    def flush(self, frame):
        if not self.itens:
            return frame
        if self._usar_pil:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            for texto, x, y, tam, cor, _negrito in self.itens:
                rgb = (cor[2], cor[1], cor[0])
                draw.text((x + 1, y + 1), texto, font=self._font(tam), fill=(0, 0, 0))
                draw.text((x, y), texto, font=self._font(tam), fill=rgb)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            for texto, x, y, tam, cor, _negrito in self.itens:
                escala = tam / 28.0
                t = _strip_accents(texto)
                org = (x, y + tam)
                cv2.putText(frame, t, org, cv2.FONT_HERSHEY_SIMPLEX, escala,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, t, org, cv2.FONT_HERSHEY_SIMPLEX, escala,
                            cor, 1, cv2.LINE_AA)
        self.itens = []
        return frame


# --------------------------------------------------------------------------- #
# Funções de desenho
# --------------------------------------------------------------------------- #
def painel(img, x, y, w, h, cor=COR_PAINEL, alpha=0.80):
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x or y2 <= y:
        return
    sub = img[y:y2, x:x2]
    overlay = np.full(sub.shape, cor, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)


def chip(img, hud, label, ativo, contagem, x, y, w=160, h=34):
    painel(img, x, y, w, h, cor=COR_PAINEL, alpha=0.85)
    cor = COR_OK if ativo else COR_OFF
    cv2.circle(img, (x + 16, y + h // 2), 6, cor, -1)
    txt = f"{label}: {contagem}" if ativo else f"{label}: --"
    hud.add(txt, x + 30, y + 8, tam=15, cor=COR_TXT if ativo else COR_TXT2)


def barra(img, x, y, w, h, frac, cor=COR_ACENTO):
    painel(img, x, y, w, h, cor=COR_BARRA_BG, alpha=0.9)
    fill = int(w * max(0.0, min(1.0, frac)))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), cor, -1)


# --------------------------------------------------------------------------- #
# MediaPipe: desenho e extração
# --------------------------------------------------------------------------- #
def desenhar_landmarks(image, results, face_full=False):
    if results.face_landmarks:
        if face_full:
            mp_draw.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
        mp_draw.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
    for mao in (results.left_hand_landmarks, results.right_hand_landmarks):
        if mao:
            mp_draw.draw_landmarks(
                image, mao, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_styles.get_default_hand_connections_style())


def extrair_keypoints(results):
    """Vetor de 1629 features (543 landmarks x 3) - o que entra na LSTM no TCC 2."""
    def bloco(lms, n):
        if lms:
            return np.array([[p.x, p.y, p.z] for p in lms.landmark]).flatten()
        return np.zeros(n * 3)
    pose = bloco(results.pose_landmarks, N_POSE)
    face = bloco(results.face_landmarks, N_FACE)
    lh = bloco(results.left_hand_landmarks, N_HAND)
    rh = bloco(results.right_hand_landmarks, N_HAND)
    return np.concatenate([pose, face, lh, rh])


# --------------------------------------------------------------------------- #
# Câmera
# --------------------------------------------------------------------------- #
def abrir_camera(index, width, height):
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    candidatos = [index] + [i for i in (0, 1, 2) if i != index]
    for idx in candidatos:
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ok, _ = cap.read()
                if ok:
                    print(f"[ok] Webcam aberta no indice {idx}.")
                    return cap
                cap.release()
    return None


# --------------------------------------------------------------------------- #
# Loop principal
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="LibrasVision - protótipo de demonstração (TCC 1)")
    ap.add_argument("--camera", type=int, default=0, help="indice da webcam (default 0)")
    ap.add_argument("--width", type=int, default=1280, help="largura da captura")
    ap.add_argument("--height", type=int, default=720, help="altura da captura")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                    help="model_complexity do Holistic (0=rapido, 2=preciso)")
    ap.add_argument("--no-mirror", action="store_true", help="desativa o espelhamento")
    args = ap.parse_args()

    print("=" * 60)
    print(" LibrasVision - protótipo de demonstração (TCC 1)")
    print(" Captura + extração de 543 landmarks via MediaPipe Holistic")
    print(" Controles: q sair | h HUD | t face | m espelho | f tela cheia | s print | espaço reset")
    print("=" * 60)

    cap = abrir_camera(args.camera, args.width, args.height)
    if cap is None:
        sys.exit("[erro] Nenhuma webcam disponivel. Verifique a conexao/permissoes da camera.")

    janela = "LibrasVision - Prototipo (TCC 1)"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, args.width, args.height)

    hud = TextHUD()
    mostrar_hud = True
    face_full = False
    espelhar = not args.no_mirror
    tela_cheia = False
    fps_hist = deque(maxlen=30)
    buffer_seq = deque(maxlen=SEQ_LEN)   # janela de 30 frames (rolling)
    n_screenshots = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=args.complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,      # mantem 468 pontos faciais (total = 543)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            t_loop = time.time()
            ok, frame = cap.read()
            if not ok:
                print("[aviso] Falha ao ler frame da camera.")
                break
            if espelhar:
                frame = cv2.flip(frame, 1)

            # --- inferencia (latencia medida apenas do MediaPipe) ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            t0 = time.time()
            results = holistic.process(rgb)
            latencia_ms = (time.time() - t0) * 1000.0

            # --- desenho dos landmarks ---
            desenhar_landmarks(frame, results, face_full=face_full)

            # --- contagens ---
            tem_pose = results.pose_landmarks is not None
            tem_face = results.face_landmarks is not None
            tem_le = results.left_hand_landmarks is not None
            tem_rd = results.right_hand_landmarks is not None
            total_vis = (N_POSE * tem_pose + N_FACE * tem_face +
                         N_HAND * tem_le + N_HAND * tem_rd)

            # --- janela temporal de 30 frames (so conta com pelo menos 1 mao) ---
            if tem_le or tem_rd:
                buffer_seq.append(extrair_keypoints(results))
            preenchido = len(buffer_seq)

            # --- FPS ---
            dt = time.time() - t_loop
            if dt > 0:
                fps_hist.append(1.0 / dt)
            fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0

            h, w = frame.shape[:2]

            if mostrar_hud:
                # Barra de acento no topo
                cv2.rectangle(frame, (0, 0), (w, 6), COR_ACENTO, -1)

                # Card titulo (topo-esquerda)
                painel(frame, 12, 16, 340, 70)
                hud.add("LibrasVision", 26, 22, tam=26, cor=COR_TXT)
                hud.add("Reconhecimento de sinais - TCC 1", 26, 56, tam=14, cor=COR_TXT2)

                # Card metricas (topo-direita)
                painel(frame, w - 252, 16, 240, 70)
                cor_lat = COR_OK if latencia_ms < 100 else COR_ACENTO
                hud.add(f"FPS: {fps:4.1f}", w - 238, 22, tam=16, cor=COR_TXT)
                hud.add(f"Latencia: {latencia_ms:5.1f} ms", w - 238, 46, tam=16, cor=cor_lat)
                hud.add("meta < 100 ms", w - 238, 68, tam=12, cor=COR_TXT2)

                # Chips de status (lateral esquerda)
                cy = 100
                chip(frame, hud, "Pose", tem_pose, N_POSE if tem_pose else 0, 12, cy)
                chip(frame, hud, "Mao Esq", tem_le, N_HAND if tem_le else 0, 12, cy + 40)
                chip(frame, hud, "Mao Dir", tem_rd, N_HAND if tem_rd else 0, 12, cy + 80)
                chip(frame, hud, "Face", tem_face, N_FACE if tem_face else 0, 12, cy + 120)

                # Total de landmarks
                painel(frame, 12, cy + 164, 180, 58)
                cor_total = COR_OK if total_vis == N_TOTAL else COR_TXT
                hud.add(f"{total_vis} / {N_TOTAL}", 26, cy + 170, tam=24, cor=cor_total)
                hud.add("landmarks 3D", 26, cy + 200, tam=12, cor=COR_TXT2)

                # Painel inferior: janela temporal + traducao
                ph = 118
                py = h - ph - 14
                painel(frame, 12, py, w - 24, ph)

                # Janela de 30 frames
                hud.add("Janela temporal (entrada da LSTM)", 28, py + 10, tam=14, cor=COR_TXT2)
                barra(frame, 28, py + 36, w - 56, 18, preenchido / SEQ_LEN)
                if preenchido >= SEQ_LEN:
                    msg = f"Janela de {SEQ_LEN} frames pronta  ->  LSTM (modulo do TCC 2)"
                    cor_msg = COR_OK
                else:
                    msg = f"Coletando frames com maos visiveis: {preenchido}/{SEQ_LEN}"
                    cor_msg = COR_TXT2
                hud.add(msg, 28, py + 58, tam=14, cor=cor_msg)

                # Traducao (placeholder honesto)
                hud.add("Traducao:", 28, py + 84, tam=16, cor=COR_TXT2)
                hud.add("[ classificacao do sinal: modulo do TCC 2 ]", 140, py + 84,
                        tam=16, cor=COR_ACENTO)

                frame = hud.flush(frame)
            else:
                hud.itens = []

            cv2.imshow(janela, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):          # q ou ESC
                break
            elif key == ord("h"):
                mostrar_hud = not mostrar_hud
            elif key == ord("t"):
                face_full = not face_full
            elif key == ord("m"):
                espelhar = not espelhar
            elif key == ord("f"):
                tela_cheia = not tela_cheia
                cv2.setWindowProperty(
                    janela, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if tela_cheia else cv2.WINDOW_NORMAL)
            elif key == ord("s"):
                nome = f"librasvision_print_{n_screenshots:02d}.png"
                cv2.imwrite(nome, frame)
                n_screenshots += 1
                print(f"[ok] Print salvo: {nome}")
            elif key == ord(" "):
                buffer_seq.clear()

    cap.release()
    cv2.destroyAllWindows()
    print("[ok] Encerrado.")


if __name__ == "__main__":
    main()
