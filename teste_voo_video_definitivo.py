import os
# FORÇA o OpenCV do Windows a não criar buffer de vídeo e abrir pacotes UDP instantaneamente
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"

import cv2
import time
import logging
import threading
from djitellopy import Tello

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

class SistemaEdgeTello:
    """
    Arquitetura Edge Tolerante a Falhas:
    Separa totalmente a telemetria/voo (UDP 8889) do stream de vídeo (UDP 11111).
    Garante que o drone pouse em segurança mesmo se a imagem oscilar no Windows.
    """
    def __init__(self):
        self.tello = Tello()
        self.voando = False
        self.executando_teste = True
        self.bateria_atual = 0

    def thread_controle_voo(self):
        """
        Thread independente para controle de voo.
        Nunca é bloqueada por atrasos na renderização de imagem.
        """
        try:
            logging.info(" [VOO] Aguardando 3 segundos para estabilizar conexão UDP...")
            time.sleep(3)

            logging.info(" [VOO] Enviando comando TAKEOFF (Decolagem)...")
            self.tello.takeoff()
            self.voando = True
            
            logging.info(" [VOO] Drone no ar! Resfriamento ativo das hélices. Mantendo voo por 5 segundos...")
            time.sleep(5)

            altura = self.tello.get_distance_tof()
            logging.info(f" [TELEMETRIA NO AR] Altura ToF: {altura} cm | Bateria: {self.bateria_atual}%")

        except Exception as e:
            logging.error(f" [VOO] Erro na rotina de voo: {e}")

        finally:
            logging.info(" [VOO] Enviando comando LAND (Pouso)...")
            try:
                self.tello.land()
                self.voando = False
                logging.info(" [VOO] Pouso finalizado com sucesso!")
            except Exception as e_land:
                logging.error(f" [VOO] Erro ao pousar: {e_land}")
            finally:
                time.sleep(1)
                self.executando_teste = False

    def executar_sistema_integrado(self):
        try:
            logging.info("Conectando ao DJI Tello...")
            self.tello.connect()
            self.bateria_atual = self.tello.get_battery()
            logging.info(f"Conexão OK! Bateria atual: {self.bateria_atual}%")

            if self.bateria_atual < 20:
                logging.warning("Bateria insuficiente (<20%). Abortando.")
                return

            # Limpa qualquer stream anterior travado na memória do drone
            try:
                self.tello.streamoff()
                time.sleep(0.5)
            except Exception:
                pass

            logging.info("Ligando câmera do drone (streamon)...")
            self.tello.streamon()
            time.sleep(1.5)

            # Usamos o leitor interno otimizado sem fila bloqueante (with_queue=False)
            frame_read = self.tello.get_frame_read(with_queue=False)

            # Inicia o voo em PARALELO (Background Thread)
            voo_thread = threading.Thread(target=self.thread_controle_voo, daemon=True)
            voo_thread.start()

            logging.info(" [VÍDEO] Abrindo janela de exibição. Pressione 'q' para pouso de emergência.")

            while self.executando_teste:
                frame = frame_read.frame

                if frame is not None and frame.size > 0:
                    frame_exibicao = cv2.resize(frame, (720, 480))
                    
                    status_txt = "VOANDO (Resfriando)" if self.voando else "NO CHAO"
                    cor_status = (0, 255, 0) if self.voando else (0, 255, 255)
                    
                    cv2.putText(
                        frame_exibicao, 
                        f"Status: {status_txt} | Bat: {self.bateria_atual}%", 
                        (20, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        cor_status, 
                        2
                    )

                    cv2.imshow("DJI Tello - Visao Aerea (Edge Computing)", frame_exibicao)
                else:
                    # Se o frame atrasar, não derruba o sistema; apenas aguarda o próximo pacote
                    time.sleep(0.01)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.warning(" [EMERGÊNCIA] Pouso manual solicitado pelo operador!")
                    if self.voando:
                        self.tello.land()
                    break

                time.sleep(0.02)

        except Exception as e:
            logging.error(f"Erro geral: {e}")

        finally:
            logging.info("Encerrando sistema...")
            self.executando_teste = False
            try:
                self.tello.streamoff()
            except Exception:
                pass
            cv2.destroyAllWindows()
            self.tello.end()
            logging.info("Sistema encerrado limpa e seguramente.")

if __name__ == "__main__":
    app = SistemaEdgeTello()
    app.executar_sistema_integrado()