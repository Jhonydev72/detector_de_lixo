import cv2
import time
import logging
import threading
from djitellopy import Tello

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

class VooComVideoTeste:
    """
    Módulo de teste integrado (RF02, RF03, RF05):
    Executa o controle de voo em paralelo com a exibição de vídeo via OpenCV.
    """
    def __init__(self):
        self.tello = Tello()
        self.voando = False
        self.executando = True

    def rotina_de_voo(self):
        """
        Executada em uma thread separada para NÃO TRAVAR o vídeo do OpenCV.
        """
        try:
            logging.info(" [VOO] Aguardando 3 segundos para estabilizar o vídeo antes de decolar...")
            time.sleep(3)

            logging.info(" [VOO] Enviando comando de decolagem (takeoff)...")
            self.tello.takeoff()
            self.voando = True
            
            logging.info(" [VOO] Drone no ar! Mantendo voo estacionário por 6 segundos...")
            time.sleep(6)

            # Exibe telemetria no meio do voo
            altura = self.tello.get_distance_tof()
            bateria = self.tello.get_battery()
            logging.info(f" [TELEMETRIA] Bateria: {bateria}% | Altura ToF: {altura} cm")

        except Exception as e:
            logging.error(f" [VOO] Erro durante a rotina de voo: {e}")

        finally:
            logging.info(" [VOO] Enviando comando de pouso (land)...")
            try:
                self.tello.land()
                self.voando = False
                logging.info(" [VOO] Pouso finalizado com segurança!")
            except Exception as e_land:
                logging.error(f" [VOO] Erro ao tentar pousar: {e_land}")
            finally:
                # Avisa o loop principal de vídeo que o teste de voo acabou
                time.sleep(2)
                self.executando = False

    def iniciar_teste_integrado(self):
        try:
            logging.info("Conectando ao DJI Tello...")
            self.tello.connect()
            bateria = self.tello.get_battery()
            logging.info(f"Conexão estabelecida! Bateria atual: {bateria}%")

            if bateria < 20:
                logging.warning("Bateria muito baixa (< 20%). Teste abortado por segurança.")
                return

            # Ativa a câmera usando o leitor de background padrão da biblioteca
            logging.info("Ligando stream de vídeo (streamon)...")
            self.tello.streamon()
            frame_read = self.tello.get_frame_read()
            time.sleep(1)

            # Inicia a rotina de voo EM PARALELO (Background Thread)
            thread_voo = threading.Thread(target=self.rotina_de_voo, daemon=True)
            thread_voo.start()

            logging.info("Janela de vídeo aberta! Pressione 'q' caso precise de POUSO DE EMERGÊNCIA.")

            # Loop Principal: Exibição contínua do vídeo
            while self.executando:
                frame = frame_read.frame

                if frame is not None and frame.size > 0:
                    # Redimensiona para exibição
                    frame_exibicao = cv2.resize(frame, (720, 480))
                    
                    # Adiciona um indicador visual de status na própria imagem
                    status_texto = "VOANDO (Resfriamento Ativo)" if self.voando else "NO CHAO"
                    cv2.putText(
                        frame_exibicao, 
                        f"Status: {status_texto} | Bat: {bateria}%", 
                        (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0) if self.voando else (0, 255, 255), 
                        2
                    )

                    cv2.imshow("DJI Tello - Voo + Video (Edge Computing)", frame_exibicao)

                # Se o operador apertar 'q', pousa imediatamente e encerra
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.warning("Interrupção solicitada! Forçando pouso imediato...")
                    if self.voando:
                        self.tello.land()
                    break

                time.sleep(0.03) # ~30 FPS

        except Exception as e:
            logging.error(f"Erro geral no sistema: {e}")

        finally:
            logging.info("Encerrando stream, janelas e conexão...")
            self.executando = False
            try:
                self.tello.streamoff()
            except Exception:
                pass
            cv2.destroyAllWindows()
            self.tello.end()
            logging.info("Teste integrado concluído.")

if __name__ == "__main__":
    app = VooComVideoTeste()
    app.iniciar_teste_integrado()