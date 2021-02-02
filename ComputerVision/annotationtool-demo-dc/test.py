import threading as th
import keyboard
import pynput.keyboard as keyboard

k_button = keyboard.Controller()
k_key = keyboard.Key


def input_key(key):
    if key == ord('s'):
        print("s")
    elif key == ord('k'):
        print('k')
    elif key == ord("u"):
        print('u')


thread1 = th.Thread(target=input_key, args=(cv2.waitKey(0), ))
thread1.daemon = True
thread1.start()
count = -99
while count != 0:

    print(count)
