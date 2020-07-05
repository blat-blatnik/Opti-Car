print('')
print('autoinfo.py helps you generate pyautogui scripts')
print('  [ctrl-c] stop the program')
print('  [space]  permanently print current info')
print('')


from pyautogui import *
from pynput.keyboard import Key, Listener

info = ''

def keypress(key):
	if (key == Key.space):
		print(info + '\n', flush=True)


with Listener(on_press=keypress) as listener:
	while True:
		x, y = position()
		r, g, b = pixel(x, y)
		info = 'mousepos [%4s %4s] rgb [%3s %3s %3s]' % (x, y, r, g, b)
		print(info, end='')
		print('\b' * len(info), end='', flush=True)