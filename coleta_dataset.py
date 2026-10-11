import cv2
import os
import time
import logging
from djitellopy import Tello

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

class ColetorDatasetEdge:
    """
    Módulo de pilotagem manual e captura de dados visuais.
    Permite voar o drone pelo teclado e capturar frames para treinamento da IA.
    """
    def __init__(self):
        self.tello = Tello()
        self.velocidade = 50 # Velocidade inicial do drone (10 a 100)
        self.pasta_dataset = "dataset_lixo"
        self.contador_fotos = 0

        # Cria a pasta para armazenar as fotos caso não exista
        if not os.path.exists(self.pasta_dataset):
            os.makedirs(self.pasta_dataset)
            logging.info(f"Pasta '{self.pasta_dataset}' criada com sucesso.")

    def iniciar_coleta(self):
        try:
            logging.info("Conectando ao DJI Tello...")
            self.tello.connect()
            logging.info(f"Conexão estabelecida! Bateria: {self.tello.get_battery()}%")

            self.tello.streamon()
            frame_read = self.tello.get_frame_read()
            time.sleep(2)

            logging.info("=== CONTROLES DE VOO ===")
            logging.info("[ T ] - Decolar (Takeoff)")
            logging.info("[ Y ] - Pousar (Land)")
            logging.info("[ W/S/A/D ] - Frente / Trás / Esquerda / Direita")
            logging.info("[ I/K ] - Subir / Descer")
            logging.info("[ J/L ] - Girar no próprio eixo (Yaw)")
            logging.info("[ X / Z ] - Aumentar / Diminuir Velocidade")
            logging.info("[ C ] - Capturar Foto (Salva no dataset)")
            logging.info("[ Q ] - Pouso de emergência e Sair")
            logging.info("========================")

            while True:
                frame = frame_read.frame
                if frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue

                frame_exibicao = cv2.resize(frame, (1280, 720))
                
                # Interface na tela atualizada com o indicador de velocidade
                texto_hud = f"Fotos: {self.contador_fotos} | Bat: {self.tello.get_battery()}% | Vel: {self.velocidade}"
                cv2.putText(frame_exibicao, texto_hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Captura de Dataset - DJI Tello", frame_exibicao)

                # Leitura do teclado (o OpenCV captura a tecla pressionada)
                tecla = cv2.waitKey(50) & 0xFF

                # Inicializa as velocidades como 0 a cada iteração
                lr, fb, ud, yv = 0, 0, 0, 0

                if tecla == ord('q'):
                    logging.warning("Encerrando programa...")
                    break
                elif tecla == ord('t'):
                    self.tello.takeoff()
                elif tecla == ord('y'):
                    self.tello.land()
                elif tecla == ord('c'):
                    # Salva a imagem
                    nome_arquivo = os.path.join(self.pasta_dataset, f"saco_lixo_{int(time.time())}.jpg")
                    cv2.imwrite(nome_arquivo, frame)
                    self.contador_fotos += 1
                    logging.info(f"Foto salva: {nome_arquivo} (Total: {self.contador_fotos})")
                
                # --- CONTROLES DE VELOCIDADE DINÂMICA ---
                elif tecla == ord('x'):
                    self.velocidade = min(100, self.velocidade + 10)
                    logging.info(f"Velocidade aumentada para: {self.velocidade}")
                elif tecla == ord('z'):
                    self.velocidade = max(10, self.velocidade - 10)
                    logging.info(f"Velocidade reduzida para: {self.velocidade}")

                # --- CONTROLES DE MOVIMENTO ---
                elif tecla == ord('w'):
                    fb = self.velocidade
                elif tecla == ord('s'):
                    fb = -self.velocidade
                elif tecla == ord('a'):
                    lr = -self.velocidade
                elif tecla == ord('d'):
                    lr = self.velocidade
                elif tecla == ord('i'):
                    ud = self.velocidade
                elif tecla == ord('k'):
                    ud = -self.velocidade
                elif tecla == ord('j'):
                    yv = -self.velocidade
                elif tecla == ord('l'):
                    yv = self.velocidade

                # Envia os comandos de movimento para o drone
                self.tello.send_rc_control(lr, fb, ud, yv)

        except Exception as e:
            logging.error(f"Erro durante a coleta: {e}")

        finally:
            logging.info("Pousando e limpando recursos...")
            try:
                self.tello.land()
            except Exception:
                pass
            self.tello.streamoff()
            cv2.destroyAllWindows()
            self.tello.end()

if __name__ == "__main__":
    coletor = ColetorDatasetEdge()
    coletor.iniciar_coleta()