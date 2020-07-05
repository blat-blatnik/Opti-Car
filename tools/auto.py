# This script automates bulk uploading to https://photoscissors.com/
# It was programmed with exactly pixel-perfect positions and colors that
# only work on my computer, so it *PROBABLY WONT WORK* on your computer.
# You have to tweak all the pixel positions and stuff to get it to work.
# You can use autoinfo.py to help you tweak everything.

from pyautogui import *
from time import sleep


PAUSE = 1.0
slp = 0.2


def waitColor(x, y, color, func=None, slptime=0.02):
	while not pixelMatchesColor(x, y, color):
		sleep(slptime)
		if func != None:
			func()
		
		
def waitNotColor(x, y, color, func=None):
	while pixelMatchesColor(x, y, color):
		sleep(0.02)
		if func != None:
			func()
		
		
def colorEq(x, y):
	return x[0] == y[0] and x[1] == y[1] and x[2] == y[2] 
		

for t in reversed(range(3 + 1)):
	print(t)
	sleep(1)
	
	
i = 0	
while True:
	
	i += 1
	print('iteration ' + str(i))

	print('  clicking on upload...')
	moveTo(950, 605, duration=0.2)
	waitColor(950, 605, (40, 96, 144))
	click(950, 605)
	sleep(slp)
	
	print('  waiting for selection window...')
	def waitForSelectionWindow():
		click(950, 605)
		sleep(0.1)
	waitColor(1300, 540, (32, 32, 32), waitForSelectionWindow)
	sleep(slp)
	
	print('  selecting previous image...')
	moveTo(1142, 567, duration=0.2)
	click(x=1142, y=567)
	waitColor(1142, 567, (119, 119, 119))
	sleep(slp)
	
	print('  deleting previous image...')
	def deletePreviousImage():
		press('delete')
		sleep(0.05)
	waitNotColor(1142, 567, (119, 119, 119), deletePreviousImage)
	sleep(slp)
	
	print('  selecting next image for upload...')
	def selectNextImage():
		doubleClick(x=1142, y=567)
		sleep(0.1)
	waitNotColor(1300, 540, (32, 32, 32), selectNextImage)
	
	print('  checking for bad request...')
	moveTo(97, 486, duration=0.2)
	startOver = False
	while True:
		if pixelMatchesColor(156, 563, (182, 40, 40), tolerance=10):
			print('!!! BAD REQUEST !!!')
			print('  skipping this image...')
			moveTo(84, 52, duration=0.2)
			sleep(slp)
			click(x=84, y=52)
			startOver = True
			break
		elif pixelMatchesColor(27, 126, (234, 54, 48)):
			print('  everything seems fine...')
			break
		else:
			sleep(1)
			
	if startOver:
		continue
	
	print('  switching to background chooser panel...')
	moveTo(1773, 176, duration=0.2)
	waitColor(1773, 176, (238, 238, 238))
	waitColor(1773, 176, (255, 255, 255), lambda: click(x=1773, y=176)) 
	sleep(slp)
	
	print('  clicking on drop down menu')
	moveTo(1899, 241, duration=0.2)
	waitColor(1899, 241, (225, 225, 225))
	waitColor(1899, 241, (229, 241, 251), lambda: click(x=1899, y=241))
	sleep(slp)
	
	print('  selecting solid color...')
	def moveBackAndForth():
		moveTo(1766, 305)
		moveTo(1765, 305)
	moveTo(1765, 305, duration=0.2)
	waitColor(1765, 305, (0, 120, 215), moveBackAndForth)
	click(x=1765, y=305)
	sleep(slp)
	
	print('  making sure solid color is applied...')
	waitColor(1770, 325, (255, 255, 255))
	sleep(slp)
	
	print('  downloading...')
	moveTo(1812, 131, duration=0.2)
	waitColor(1812, 131, (40, 96, 144))
	sleep(slp)
	click(x=1812, y=131)
	
	print('  going back...')
	moveTo(15, 53, duration=0.2)
	waitColor(15, 53, (213, 213, 213))
	click(x=15, y=53)
	waitColor(20, 180, (82, 186, 213), lambda: click(x=15, y=53), 1.0)