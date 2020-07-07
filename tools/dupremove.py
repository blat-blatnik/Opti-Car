# How to use:
#
# $ pythong dupremove.py SOME_DIRECTORY
#
# Where SOME_DIRECTORY is the directory in which you have duplicates
# for example:
#
# $ python3 dupremove.py "../Neural Nets/cars3"
#
# This will find all files in the directory whose MD5 hashsum is exactly
# the same and move the duplicate files into a 'Duplicates' directory.
# So for the above example the duplicate files would be moved to
# ../Neural Nets/cars3/Duplicates/

from hashlib import md5
from shutil import move
from glob import glob
from sys import argv
from os import mkdir

def image_hash(filename):
	with open(filename, 'rb') as f:
		return md5(f.read()).hexdigest()
			
if __name__ == '__main__':
	dir = argv[1]
	pattern = dir + '/*.*'	
	files = glob(pattern)
	dupdir = dir + '/Duplicates'
	
	try:
		mkdir(dupdir)
	except OSError:
		idontcare = 0

	print('checking all %d files matching %s..' % (len(files), pattern))
	print('duplicates will be moved to %s' % dupdir)
	
	hashset = set()
	numdups = 0
	for file in files:
		hash = image_hash(file)
		if hash in hashset:
			numdups += 1
			print(' found duplicate: %s' % file)
			move(file, dupdir)
		else:
			hashset.add(hash)
			
	print('%d duplicates moved to %s' % (numdups, dupdir))