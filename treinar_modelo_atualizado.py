from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

def iniciar_treinamento_gpu():
    logging.info("Carregando modelo base YOLOv8 Nano...")
    modelo = YOLO('yolov8n.pt') 

    logging.info("Iniciando treinamento multiclasse acelerado na RTX 2050 (GPU 0)...")
    
    resultados = modelo.train(
        # COLOQUE O CAMINHO ABSOLUTO DO SEU YAML AQUI
        data='C:/Users/JoãoF/Desktop/drone/dataset_multiclasse/data.yaml', 
        epochs=50,
        imgsz=640, 
        plots=True,
        device=0,
        # AS DUAS LINHAS ABAIXO PROTEGEM O SEU MODELO ANTIGO
        project='modelos_tcc',      # Cria uma pasta principal nova
        name='yolo_multiclasse_v1'  # Salva o best.pt isolado aqui dentro
    )
    
    logging.info("Treinamento finalizado! Pesos salvos na pasta 'modelos_tcc/yolo_multiclasse_v1/weights/best.pt'")

if __name__ == "__main__":
    iniciar_treinamento_gpu()