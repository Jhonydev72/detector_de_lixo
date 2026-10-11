from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

def iniciar_treinamento_gpu():
    logging.info("Carregando modelo base YOLOv8 Nano...")
    modelo = YOLO('yolov8n.pt') 

    logging.info("Iniciando treinamento acelerado na RTX 2050 (GPU 0)...")
    
    # O parâmetro device=0 força o uso exclusivo da sua placa NVIDIA
    resultados = modelo.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        plots=True,
        device=0  
    )
    
    logging.info("Treinamento finalizado! Pesos salvos em 'runs/detect/train/weights/best.pt'")

if __name__ == "__main__":
    iniciar_treinamento_gpu()