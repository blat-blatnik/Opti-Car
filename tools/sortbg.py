# sort all *JPEG* images in the *CURRENT* directory into images that
# have a completely white background (no background) and images that
# have some more complex background.

from glob import glob
from os import mkdir
from PIL import Image
from shutil import move
	
	
def hasBackground(image):
	width, height = image.size
	
	for y in range(height):
		if y == 0 or y == height - 1:
			step = 1
		else:
			step = width - 1
		for x in range(0, width, step):
			p = image.getpixel((x, y))
			if p[0] < 240 or p[1] < 240 or p[2] < 240:
				return True
				
	return False
	
	
try:
	mkdir('Background')
except OSError:
	idontcare=0
try:
	mkdir('No_Background')
except OSError:
	idontcare=0
	

i = 0	
for file in glob('*.jpg'):
	i += 1
	print('moving %d' % i)
	im = Image.open(file)
	
	if hasBackground(im):
		move(file, 'Background')
	else:
		move(file, 'No_Background')

		
print('')
print('done!')