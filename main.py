# pyrefly: ignore [missing-import]
from djitellopy import tello

drone = tello.Tello()


drone.connect()

drone.takeoff()

drone.rotation_clockwise(360)

drone.land()

drone.end()

