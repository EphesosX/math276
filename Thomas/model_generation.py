import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
from matplotlib import pyplot as plt

def generate_data(noise=False, nums=None):
	K.set_image_dim_ordering('th')
	# 4. Load pre-shuffled MNIST data into train and test sets
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	 
	# 5. Preprocess input data

	X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
	X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	# 6. Preprocess class labels
	Y_train = np_utils.to_categorical(y_train, 10)
	Y_test = np_utils.to_categorical(y_test, 10)
	if nums:
		# Filter out only training and test examples corresponding to the given digits
		X_train_nums = []
		X_test_nums = []
		Y_train_nums = []
		Y_test_nums = []
		
		for num in nums:
			X_train_num = X_train[Y_train[:,num]==1,:,:,:]
			X_test_num = X_test[Y_test[:,num]==1,:,:]
			Y_train_num = Y_train[Y_train[:,num]==1]
			Y_test_num = Y_test[Y_test[:,num]==1]

			X_train_nums.append(X_train_num)
			X_test_nums.append(X_test_num)
			Y_train_nums.append(Y_train_num)
			Y_test_nums.append(Y_test_num)

		X_train = np.concatenate(X_train_nums)
		X_test = np.concatenate(X_test_nums)
		Y_train = np.concatenate(Y_train_nums)
		Y_test = np.concatenate(Y_test_nums)
	if noise:
		X_train = np.array([salt_pepper(X_train[i,0,:,:]) for i in range(X_train.shape[0])])
		X_train = np.reshape(X_train,[60000,1,28,28])
		X_test = np.array([salt_pepper(X_test[i,0,:,:]) for i in range(X_test.shape[0])])
		X_test = np.reshape(X_test,[10000,1,28,28])
	return X_train, X_test, Y_train, Y_test

def generate_model(noise=False, force_retrain=False, nums=None, name=None):
	if not force_retrain:
		try:
			if name:
				model = load_model(name)
			elif noise:
				model = load_model('keras_mnist_model_noisev1.h5')
			else:
				model = load_model('keras_mnist_modelv1.h5')
			return model
		except:
			pass

	np.random.seed(123)  # for reproducibility
	 
	X_train, X_test, Y_train, Y_test = generate_data(noise, nums)
	 
	# 7. Define model architecture
	model = Sequential()
	 
	model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
	model.add(Convolution2D(32, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	 
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	 
	# 8. Compile model
	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	 
	# 9. Fit model on training data
	model.fit(X_train, Y_train, 
	          batch_size=32, nb_epoch=10, verbose=1)
	 
	# 10. Evaluate model on test data
	score = model.evaluate(X_test, Y_test, verbose=0)
	if name:
		model.save(name)
	elif noise:
		model.save('keras_mnist_model_noisev1.h5')
	else:
		model.save('keras_mnist_modelv1.h5')
	return model


def floats_to_rgb(float_array):
    max_val = np.max(float_array)
    min_val = np.min(float_array)
    avg = (max_val + min_val)/2.0
    avg_array = np.ones(float_array.shape)*avg
    zero_array = np.zeros(float_array.shape)
    min_array = np.ones(float_array.shape)*min_val
    red = (np.where(float_array>avg, float_array, avg_array)-avg_array)/(max_val-avg)
#     blue = (avg_array-np.where(float_array<avg, float_array,avg_array))/(avg-min_val)
    blue = zero_array
#     green = (avg-min_val-abs(float_array-avg_array))/(avg-min_val)
    green = zero_array
    return np.dstack([red, green, blue])
   
def plot_model(model):
	# test = np.array([2.0,4.0,3.4,6.4])
	# print([x for x in floats_to_rgb(test)])
	weights = np.abs(np.array(model.layers[-1].get_weights()[0]))
	thresh = np.percentile(weights, 94)
	weights[weights<thresh] = 0
	colors = floats_to_rgb(weights)

	# thresh = np.average(np.abs(np.array(model.layers[-1].get_weights()[0])))*2

	x0 = np.linspace(0,100,128)
	x1 = np.linspace(0,100,10)
	y0 = 0
	y1 = 100

	for x, row in enumerate(colors):
	    for y, rgb in enumerate(row):
	        if abs(np.array(model.layers[-1].get_weights()[0])[x,y])>thresh:
	            plt.plot([x0[x],x1[y]],[y0,y1], color=rgb)

	plt.show()

def salt_pepper(im,ps=.1,pp=.1):
    im1=im[:,:].copy()
    n,m=im1.shape
    for i in range(n):
        for j in range(m):
            b=np.random.uniform()
            if b<ps:
                im1[i,j]=0
            elif b>1-pp:
                im1[i,j]=1
    noisy_im=im1
    return noisy_im