import cv2
import time
import logging
from djitellopy import Tello

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

class AnalisadorVideoTeste:
    """
    Módulo de validação de recepção de frames na estação base (Edge)[cite: 1].
    Otimizado para o Windows: evita timeout do OpenCV e superaquecimento do VANT[cite: 1].
    """
    def __init__(self):
        self.tello = Tello()

    def testar_stream_ao_vivo(self):
        cap = None
        try:
            logging.info("Conectando ao DJI Tello...")
            self.tello.connect()
            logging.info(f"Conexão OK! Bateria atual: {self.tello.get_battery()}%")

            # Desliga stream anterior para limpar o buffer do drone
            try:
                self.tello.streamoff()
                time.sleep(0.5)
            except Exception:
                pass

            logging.info("Ativando stream de vídeo da câmera (streamon)...")
            self.tello.streamon()
            time.sleep(1.5)

            # Uso do '@' antes do IP/porta é essencial no FFMPEG do Windows
            # para bind de portas UDP sem bloqueio de socket
            endereco_udp = "udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=50000000"
            logging.info("Abrindo captura de vídeo na porta UDP 11111...")
            
            cap = cv2.VideoCapture(endereco_udp, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                logging.error("Não foi possível abrir a porta de vídeo UDP.")
                return

            logging.info("Stream ativo! Pressione 'q' para fechar ou 's' para salvar frame[cite: 1].")

            tentativas_vazias = 0

            while True:
                ret, frame = cap.read()

                if not ret or frame is None:
                    tentativas_vazias += 1
                    if tentativas_vazias > 50:
                        logging.warning("Sem sinal de vídeo. Verifique se há cabos Ethernet ativos no PC.")
                        break
                    time.sleep(0.05)
                    continue

                tentativas_vazias = 0  # Reseta o contador ao receber frame válido

                # Redimensiona para exibição fluida na estação base[cite: 1]
                frame_exibicao = cv2.resize(frame, (720, 480))
                cv2.imshow("DJI Tello - Stream de Video (Edge Computing)", frame_exibicao)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord('q'):
                    logging.info("Encerramento solicitado pelo operador[cite: 1].")
                    break
                elif tecla == ord('s'):
                    cv2.imwrite("amostra_tello.jpg", frame)
                    logging.info("Frame salvo como 'amostra_tello.jpg'!")

        except Exception as e:
            logging.error(f"Erro no fluxo de vídeo: {e}")

        finally:
            logging.info("Encerrando conexões...")
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            try:
                # Envia streamoff com tratamento silencioso para evitar log de WinError 6
                self.tello.streamoff()
            except Exception:
                pass
            logging.info("Teste de vídeo encerrado.")

if __name__ == "__main__":
    app = AnalisadorVideoTeste()
    app.testar_stream_ao_vivo()