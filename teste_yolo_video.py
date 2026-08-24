import cv2
import time
import logging
from djitellopy import Tello
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

class AnalisadorIAEdge:
    """
    Integração do modelo YOLOv8 para detecção de objetos em tempo real (RF06, RF07).
    """
    def __init__(self):
        self.tello = Tello()
        logging.info("Carregando modelo YOLOv8 (versão Nano para maior FPS)...")
        # Baixa automaticamente o modelo pré-treinado na primeira execução
        self.modelo = YOLO('yolov8n.pt') 

    def testar_deteccao_ao_vivo(self):
        try:
            logging.info("Conectando ao drone...")
            self.tello.connect()
            logging.info(f"Bateria: {self.tello.get_battery()}%")

            self.tello.streamon()
            time.sleep(2) # Aguarda estabilização do hardware
            frame_read = self.tello.get_frame_read(with_queue=False)

            logging.info(">>> STREAM IA INICIADO! Pressione 'q' para sair. <<<")

            while True:
                frame = frame_read.frame

                if frame is not None and frame.size > 0:
                    # Redimensiona para manter o desempenho
                    frame_processamento = cv2.resize(frame, (720, 480))
                    
                    # --- INFERÊNCIA YOLOv8 ---
                    # verbose=False evita poluir o terminal com logs a cada frame
                    resultados = self.modelo(frame_processamento, verbose=False)
                    
                    # O Ultralytics possui um método nativo (plot) que já desenha 
                    # as bounding boxes e o confianca_ia[cite: 1] na imagem
                    frame_anotado = resultados[0].plot()

                    # Mostra a imagem com a Inteligência Artificial atuando
                    cv2.imshow("DJI Tello - Visao Computacional (YOLOv8)", frame_anotado)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Encerramento solicitado.")
                    break
                
                time.sleep(0.01)

        except Exception as e:
            logging.error(f"Erro no processamento de IA: {e}")

        finally:
            logging.info("Limpando recursos...")
            try:
                self.tello.streamoff()
            except Exception:
                pass
            cv2.destroyAllWindows()
            self.tello.end()

if __name__ == "__main__":
    app = AnalisadorIAEdge()
    app.testar_deteccao_ao_vivo()