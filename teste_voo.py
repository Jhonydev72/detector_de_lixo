import time
import logging
from djitellopy import Tello

# Configuração básica de log para acompanhamento no terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

class ControllerVoo:
    """
    Classe responsável pelo controle físico e telemetria do DJI Tello.
    Representa o núcleo operacional de hardware da arquitetura Edge.
    """
    def __init__(self):
        self.tello = Tello()
        self.conectado = False

    def conectar(self) -> bool:
        """Estabelece conexão UDP com o drone e verifica o nível de bateria."""
        try:
            logging.info("Iniciando conexão com o DJI Tello...")
            self.tello.connect()
            self.conectado = True
            
            bateria = self.tello.get_battery()
            logging.info(f"Conexão estabelecida com sucesso! Bateria atual: {bateria}%")
            
            # Trava de segurança: evita decolagem com bateria crítica
            if bateria < 20:
                logging.warning("Bateria abaixo de 20%. Operação de voo abortada por segurança.")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Falha ao conectar com o drone: {e}")
            return False

    def exibir_telemetria_basica(self):
        """Coleta e exibe dados básicos de telemetria do voo."""
        try:
            bateria = self.tello.get_battery()
            altura = self.tello.get_distance_tof()  # Altura em cm via sensor ToF
            temp = self.tello.get_temperature()
            
            logging.info("--- TELEMETRIA DO VOO ---")
            logging.info(f"Bateria: {bateria}% | Altura estimada: {altura} cm | Temperatura: {temp}°C")
            logging.info("-------------------------")
        except Exception as e:
            logging.error(f"Erro ao ler telemetria: {e}")

    def teste_voo_basico(self):
        """
        Executa o fluxo da Sprint 1: conectar, decolar, manter voo estacionário,
        ler telemetria e pousar com segurança.
        """
        if not self.conectar():
            logging.error("Não foi possível iniciar o teste de voo.")
            return

        try:
            logging.info("Enviando comando de decolagem (takeoff)...")
            self.tello.takeoff()
            
            logging.info("Drone em voo estacionário. Aguardando estabilização (5 segundos)...")
            time.sleep(5)
            
            # Leitura de telemetria durante o voo
            self.exibir_telemetria_basica()
            
            logging.info("Mantendo posição por mais 3 segundos...")
            time.sleep(3)

        except Exception as e:
            logging.critical(f"Erro inesperado durante o voo: {e}")
            
        finally:
            # O bloco finally garante que o comando de pouso será enviado mesmo se houver erro
            logging.info("Enviando comando de pouso (land)...")
            try:
                self.tello.land()
                logging.info("Pouso finalizado com segurança.")
            except Exception as e_land:
                logging.error(f"Erro ao tentar pousar: {e_land}")
            finally:
                self.tello.end()
                logging.info("Conexão encerrada.")

if __name__ == "__main__":
    logging.info("=== INICIANDO TESTE DE VOO - SPRINT 1 ===")
    controlador = ControllerVoo()
    controlador.teste_voo_basico()