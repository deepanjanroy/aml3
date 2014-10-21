import os
import numpy as np
from scipy import misc
import shutil
import subprocess

def save_as_png(img, file_name):
	''' Given a (48x48) image in the form of a (2304x1) vector, this method
	will save it as a png to the given location.
	'''
	misc.imsave(file_name, np.reshape(img, (48,48)))
	
	
def load_from_png(file_name):
	''' Returns the image from the given file_name in a (2304x1) vector.
	'''
	img = misc.imread(file_name)
	return np.reshape(img, (2304,))
	
def create_image_rotations(img, interval):
	''' Given an image vector (2304x1), this method will rotate it 360 
	degrees with the given interval. Returns a list of (2304x1) arrays 
	containing the original image followed by its rotations.
	'''
	# Create a working directory.
	os.mkdir('temp_rotations')
	os.chdir('temp_rotations')
	# Save the original image.
	ret_images = []
	ret_images.append(img)
	save_as_png(img, 'orig.png')
	# Rotate the image between 0 and 360.
	for i in xrange(0, 359, interval):
		subprocess.call('convert orig.png -virtual-pixel Tile -distort SRT ' + str(i) + ' temp_'+str(i)+'.png', shell=True)
		im = load_from_png('temp_'+str(i)+'.png')
		ret_images.append(im)
	
	# Return to the original directory and clean temporary files.
	os.chdir('../')
	shutil.rmtree('temp_rotations')
	return ret_images
	
	
