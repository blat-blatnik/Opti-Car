from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sys import argv
from cv2 import *
from os import listdir, path
import matplotlib.pyplot as plt
import numpy as np


# model hyperparameters
# ---------------------
RNGSeed       = 42     # for reproducability
imageWidth    = 150    # these should probably be kept square - good values larger than 100
imageHeight   = 150
grayscale     = True   # train with a color image, or convert to grayscale?
learningRate  = 0.001  # model learning rate for Adam - good values around 0.001
batchSize     = 64     # batch size - set as high as possible before running out of memory
augmentFlip   = False  # augment data by flipping images vertically, producing 2 training images for every 1 image in the dataset
augmentRotate = 0      # augment data by rotating images by random angle between (-augmentRotate, +augmentRotate) degrees. set to 0 for no augmentation


#
# =============================================================
#


# train a new model using images from the given imagePath, for a given numEpochs
# when training is done a plot of the loss over every epoch is shown, and the model
# is saved to the current directory as 'model.h5'
def trainModel(imagePath, numEpochs):
	print('')
	print('loading images...')
	print('')

	images = []
	prices = []
	for file in tqdm(listdir(imagePath)):
		MSRP = file.split('_')[3]
		try:
			MSRP = int(MSRP)
		except:
			MSRP = -1

		if MSRP >= 0:
			file = path.join(imagePath, file)
			img = loadImage(file)
			images.append(img)
			prices.append(MSRP)

			if augmentFlip:
				images.append(flip(img, 1))
				prices.append(MSRP)

			if augmentRotate != 0:
				height = img.shape[0]
				width = img.shape[1]
				angle = np.random.uniform(-augmentRotate, augmentRotate)
				rotation = getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
				rotated = warpAffine(img, rotation, img.shape[1::-1], flags=INTER_LANCZOS4, borderMode=BORDER_CONSTANT, borderValue=white)
				images.append(rotated)
				prices.append(MSRP)

	images = np.array(images)
	prices = np.array(prices)
	images = images.reshape(len(images), imageWidth, imageHeight, colorChanels)

	split = train_test_split(prices, images, test_size=0.10, random_state=RNGSeed)
	trainPrices, testPrices, trainImages, testImages = split

	maxPrice = prices.max()
	trainPrices = trainPrices / maxPrice
	testPrices = testPrices / maxPrice
	
	print('')
	print('max price is %g' % maxPrice)
	print('all predictions will be scaled as a fraction of max price')
	print('')

	model = Sequential([
		Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=inputShape),
		MaxPooling2D((2, 2)),
		Dropout(0.25),
		Conv2D(64, kernel_size=(3, 3), activation='relu'),
		MaxPooling2D((2, 2)),
		Dropout(0.25),
		Conv2D(128, kernel_size=(3, 3), activation='relu'),
		Dropout(0.4),
		Flatten(),
		Dense(128, activation='relu'),
		Dropout(0.3),
		Dense(1, activation='linear')])

	model.compile(optimizer=Adam(learningRate), loss='mean_absolute_percentage_error')

	print('')
	print('training model...')
	print('')

	training = model.fit(trainImages, trainPrices, 
		validation_data=(testImages, testPrices),
		epochs=numEpochs, batch_size=batchSize)

	plt.plot(training.history['loss'], '--')
	plt.plot(training.history['val_loss'], '-')
	plt.title('Model loss function')
	plt.xlabel('epoch')
	plt.ylabel('absolute error (%)')
	plt.legend(['train', 'test'], loc='upper right')
	plt.savefig('loss.pdf')
	plt.show()
		
	print('')
	print('final model evaluation...')
	print('')

	preds = model.predict(testImages)
	diff = preds.flatten() - testPrices
	percentDiff = (diff / testPrices) * 100
	absPercentDiff = np.abs(percentDiff)

	print('mean   price difference = %g' % np.mean(absPercentDiff))
	print('stddev price difference = %g' % np.std(absPercentDiff))

	print('')
	print('saving model... ', end='')
	model.save('model.h5')
	print('DONE :)')
	print('')


# loads a model from 'model.h5' located in the current directory (crashes if this doesn't exist)
# and then uses this model to predict the prices of all images in the given path
# note that the prices are predicted on a scale relative to the max price of the prices the model was trained on
def predictPrices(imagePath):
	print('')
	print('loading model...')
	print('')
	model = load_model('model.h5')

	print('')
	print('predicting...')
	print('')
	for file in listdir(imagePath):
		img = np.array([loadImage(file)])
		img = img.reshape(1, imageWidth, imageHeight, colorChanels)
		predicted = model.predict(img)
		print('%s predicted price = %g' % (file, predicted[0]))


def loadImage(file):
	file = path.join(imagePath, file)
	if grayscale:
		img = imread(file, IMREAD_GRAYSCALE)
		height, width = img.shape
	else:
		img = imread(file)
		height, width, _ = img.shape

	maxdim = np.max([width, height])
	padx = (maxdim - width) / 2
	pady = (maxdim - height) / 2
	padLeft   = int(np.floor(padx))
	padRight  = int(np.ceil(padx))
	padTop    = int(np.floor(pady))
	padBottom = int(np.ceil(pady))
	img = copyMakeBorder(img, padTop, padBottom, padLeft, padRight, BORDER_CONSTANT, value=white)
	img = resize(img, (imageWidth, imageHeight), interpolation=INTER_LANCZOS4) / 255.0
	return img


#
# =============================================================
#


colorChanels = 1 if grayscale else 3
white = 255 if grayscale else (255, 255, 255)
inputShape = (imageWidth, imageHeight, colorChanels)
np.random.seed(RNGSeed)


if len(argv) < 2:
	print('usage')
	print('  train    $ python3 %s train DATASET-DIR NUM-EPOCHS' % argv[0])
	print('  predict  $ python3 %s IMAGE-DIR' % argv[0])
	exit()

imagePath = argv[1]
if imagePath == 'train':
	trainModel(argv[2], int(argv[3]))
else:
	predictPrices(imagePath)
