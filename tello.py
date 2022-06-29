from djitellopy import Tello
from time import sleep

def run_tello():
    tello = Tello()

    tello.connect()

    # tello.takeoff()
    # tello.move_left(100)
    # tello.rotate_ccw(90)
    # tello.move_forward(100)
    # tello.land()

    tello.get_battery()

    tello.end()

def main():
    run_tello()
    

if __name__ == '__main__':
    main()
