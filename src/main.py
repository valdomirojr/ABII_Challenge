from djitellopy import Tello
from utils import cartesian_to_polar
import time

coordenada_objetivo = ((1, 3), 60)




modulo, angulo = cartesian_to_polar(coordenada_objetivo[0])
print(angulo)
print(modulo)

tello = Tello()
tello.connect()
tello.takeoff()
time.sleep(1)
tello.rotate_clockwise(angulo)
time.sleep(1)
tello.move_forward(modulo)
time.sleep(1)
tello.rotate_clockwise(coordenada_objetivo[1])


tello.land()