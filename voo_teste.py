from djitellopy import tello

def voo_teste(drone):
    drone.connect()
    drone.takeoff()
    drone.rotation_clockwise(360)
    drone.land()
    drone.end()

drone = tello.Tello()

voo_teste(drone)

def main():

    drone.tello.Tello()
    voo_teste(drone)
if __name__ == "__main__":
    main()
    
