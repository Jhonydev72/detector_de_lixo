import cv2
from djitellopy import tello

def processo_video(drone):
    while True:
        frame = drone.get_frame_read().frame
        cv2.imshow("Drone Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    drone.end()
    cv2.destroyAllWindows()
    drone.end()
def main():
    drone = tello.Tello()
    drone.connect()
    drone.streamon()
    processo_video(drone)

if __name__ == "__main__":
    main()
    


    