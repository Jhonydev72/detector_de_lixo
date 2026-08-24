import os
# Otimizações para evitar que o FFmpeg trave placas Wi-Fi no Windows
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,rtp,udp|fflags;nobuffer|flags;low_delay|max_delay;500000"

import cv2
import time
import logging
from djitellopy import Tello

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

class ValidadorVideoPuro:
    """
    Módulo Edge Otimizado para Windows:
    Contorna falhas no driver de rede lendo o stream UDP 11111 diretamente
    via OpenCV sem utilizar as threads problemáticas da biblioteca djitellopy.
    """
    def __init__(self):
        self.tello = Tello()

    def testar_camera_sem_queda_wifi(self):
        cap = None
        try:
            logging.info("Conectando ao DJI Tello...")
            self.tello.connect()
            logging.info(f"Conexão OK! Bateria: {self.tello.get_battery()}%")

            # Desativa streams anteriores se o drone ficou travado
            try:
                self.tello.streamoff()
                time.sleep(1)
            except Exception:
                pass

            logging.info("Enviando comando STREAMON para ligar a câmera...")
            self.tello.streamon()
            
            # ESSENCIAL: Aguarda 3 segundos para o hardware da câmera do drone aquecer e transmitir
            logging.info("Aguardando estabilização do fluxo de vídeo H.264...")
            time.sleep(3.0)

            # Lendo direto da porta UDP do PC sem passar pelo parser do djitellopy
            udp_video_url = "udp://0.0.0.0:11111"
            logging.info(f"Conectando o OpenCV ao endereço: {udp_video_url}")
            
            cap = cv2.VideoCapture(udp_video_url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                logging.error("O OpenCV não conseguiu abrir a porta 11111. Verifique se outro app está usando a porta.")
                return

            logging.info(">>> STREAM DE VÍDEO CONECTADO! Pressione 'q' na janela de vídeo para sair. <<<")

            erros_consecutivos = 0

            while True:
                sucesso, frame = cap.read()

                if not sucesso or frame is None:
                    erros_consecutivos += 1
                    if erros_consecutivos > 50:
                        logging.warning("Sinal do vídeo oscilando. Tentando recuperar frame...")
                        erros_consecutivos = 0
                    time.sleep(0.02)
                    continue

                erros_consecutivos = 0

                # Redimensiona para 720x480 para exibição fluida
                frame_redimensionado = cv2.resize(frame, (720, 480))
                cv2.imshow("DJI Tello - Conexao Direta UDP (Edge)", frame_redimensionado)

                # Se o operador pressionar 'q', encerra
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Encerramento solicitado pelo operador.")
                    break

        except Exception as e:
            logging.error(f"Erro inesperado no teste de câmera: {e}")

        finally:
            logging.info("Encerrando captura e liberando recursos...")
            if cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            try:
                self.tello.streamoff()
                self.tello.end()
            except Exception:
                pass
            logging.info("Teste finalizado com segurança.")

if __name__ == "__main__":
    app = ValidadorVideoPuro()
    app.testar_camera_sem_queda_wifi()