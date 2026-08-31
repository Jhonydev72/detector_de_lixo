import cv2
import time
import logging
from djitellopy import Tello
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

class SistemaInspecaoManual:
    """
    Integração final: Controle de voo + Inferência YOLOv8 + Odometria (Mock GPS).
    Atende aos requisitos de detecção e rastreamento espacial do TCC.
    """
    def __init__(self):
        self.tello = Tello()
        self.velocidade = 50 
        
        logging.info("Carregando o modelo treinado (best.pt)...")
        # Caminho gerado no seu último treinamento
        caminho_modelo = r"runs\detect\train-6\weights\best.pt"
        self.modelo = YOLO(caminho_modelo)

        # --- VARIÁVEIS DE ODOMETRIA E GPS MOCK ---
        self.lat_inicial = -23.507830
        self.lon_inicial = -46.671310
        self.pos_x_cm = 0.0
        self.pos_y_cm = 0.0
        self.ultimo_tempo = time.time()

    def iniciar_inspecao(self):
        try:
            logging.info("Conectando ao DJI Tello...")
            self.tello.connect()
            bateria = self.tello.get_battery()
            logging.info(f"Conexão estabelecida! Bateria: {bateria}%")

            if bateria < 15:
                logging.warning("Bateria muito baixa. Troque antes de voar.")
                return

            self.tello.streamon()
            frame_read = self.tello.get_frame_read(with_queue=False)
            time.sleep(2)

            logging.info("=== CONTROLES DE VOO E INSPEÇÃO ===")
            logging.info("[ T ] - Decolar  |  [ Y ] - Pousar")
            logging.info("[ W/S/A/D ] - Mover no plano horizontal")
            logging.info("[ I/K ] - Subir / Descer")
            logging.info("[ J/L ] - Girar no próprio eixo (Yaw)")
            logging.info("[ X / Z ] - Aumentar / Diminuir Velocidade")
            logging.info("[ Q ] - Pouso de emergência e Sair")
            logging.info("===================================")

            # Reseta o cronômetro antes de entrar no loop
            self.ultimo_tempo = time.time()

            while True:
                frame = frame_read.frame
                if frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue

                # --- 1. CÁLCULO DE ODOMETRIA E GPS ---
                tempo_atual = time.time()
                delta_t = tempo_atual - self.ultimo_tempo
                self.ultimo_tempo = tempo_atual

                # Obtém a velocidade inercial instantânea em cm/s
                vx = self.tello.get_speed_x()
                vy = self.tello.get_speed_y()

                # Deslocamento (cm) = Velocidade * Tempo
                self.pos_x_cm += vx * delta_t
                self.pos_y_cm += vy * delta_t

                # Conversão matemática: 1 metro de deslocamento ~= 0.000009 graus geográficos
                lat_atual = self.lat_inicial + ((self.pos_x_cm / 100.0) * 0.000009)
                lon_atual = self.lon_inicial + ((self.pos_y_cm / 100.0) * 0.000009)

                # --- 2. PROCESSAMENTO DE IMAGEM E IA ---
                frame_processamento = cv2.resize(frame, (720, 480))
                resultados = self.modelo.predict(frame_processamento, conf=0.60, verbose=False)
                frame_anotado = resultados[0].plot()

                # --- 3. HUD (INTERFACE NA TELA) ---
                # Linha superior: Telemetria básica
                texto_hud_1 = f"Bat: {self.tello.get_battery()}% | Vel: {self.velocidade} | IA: Ativa"
                cv2.putText(frame_anotado, texto_hud_1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Linha inferior: GPS Simulado (Odometria)
                texto_hud_2 = f"LAT: {lat_atual:.6f} | LON: {lon_atual:.6f}"
                cv2.putText(frame_anotado, texto_hud_2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow("Inspecao Aerea - DJI Tello + YOLOv8 (Edge)", frame_anotado)

                # --- 4. CONTROLE PELO TECLADO ---
                tecla = cv2.waitKey(30) & 0xFF
                lr, fb, ud, yv = 0, 0, 0, 0

                if tecla == ord('q'):
                    logging.warning("Encerrando missão de inspeção...")
                    break
                elif tecla == ord('t'):
                    self.tello.takeoff()
                elif tecla == ord('y'):
                    self.tello.land()
                elif tecla == ord('x'):
                    self.velocidade = min(100, self.velocidade + 10)
                elif tecla == ord('z'):
                    self.velocidade = max(10, self.velocidade - 10)
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

                self.tello.send_rc_control(lr, fb, ud, yv)

        except Exception as e:
            logging.error(f"Erro crítico durante a operação: {e}")

        finally:
            logging.info("Pousando e encerrando sistemas...")
            try:
                self.tello.land()
            except Exception:
                pass
            self.tello.streamoff()
            cv2.destroyAllWindows()
            self.tello.end()

if __name__ == "__main__":
    sistema = SistemaInspecaoManual()
    sistema.iniciar_inspecao()