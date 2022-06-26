from djitellopy import Tello
from time import sleep

def run_tello():
    tello = Tello()

    tello.connect()

    tello.takeoff()
    tello.move_left(100)
    tello.rotate_ccw(90)
    tello.move_forward(100)
    tello.land()

    tello.end()

def main():
    #run_tello()
    tello.get_battery()

if __name__ == '__main__':
    main()
