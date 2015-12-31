#-*- coding: utf_8 -*-
import matplotlib
import cPickle
import numpy as np
import os
import sys
import keras
import librosa
from scipy.misc import imsave

import auralise

''' 2015-09-28 Keunwoo Choi
- [0] load cnn weights that is learned from keras (http://keras.io).
- [1] load a song file (STFT).
---[1-0] log_amplitude(abs(STFT))
-- [1-1] feed-forward the spectrogram,
--- [1-1-0]using NOT theano, just scipy and numpy on CPU. Thus it's slow.
-- [1-2] then 'deconve' using switch matrix for every convolutional layer.
-- [1-3] 'separate' or 'auralise' the deconved spectrogram using phase information of original signal.
- [2]listen up!

'''
def buildConvNetModel(numFr, len_freq, learning_rate=0.0005, model_name = 'vggnet5', dropout_values=[0, 0, 0.25, 0, 0.25], optimiser=None):
	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.convolutional import Convolution2D, MaxPooling2D
	from keras.layers.normalization import LRN2D
	from keras.optimizers import RMSprop, SGD
	''' Keras layer - input argumetns should be updated
	'''
	final_freq_num = len_freq
	final_frame_num = numFr

	model = Sequential()
	num_layers = 5

	image_patch_sizes = [[3,3]] * num_layers
	pool_sizes = [(2,2)] * num_layers
		
	num_stacks = [48, 48, 48, 48, 48]
	num_channel_input = 1
	
	for i in xrange(num_layers):
		if i == 0:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
												border_mode='same', 
												input_shape=(num_channel_input, len_freq, num_fr)
												activation='relu'))
		else:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
												border_mode='same', 
												activation='relu'))
			final_freq_num = final_freq_num / pool_sizes[i][0]
			final_frame_num = final_frame_num / pool_sizes[i][1]
		model.add(MaxPooling2D(poolsize=pool_sizes[i]))
		if dropout_values[i] is not 0:
			model.add(Dropout(dropout_values[i]))
		model.add(LRN2D())

	model.add(Flatten())
	model.add(Dense(512, init='normal', activation='relu'))
	model.add(Dropout(0.5))
	### [END OF METHOD 2]
	'''end of method 2'''
	model.add(Dense(256, init='normal', activation='relu'))
	model.add(Dropout(0.5))

	# Decision layer
	model.add(Dense(3, init='normal')) #no activation. here I assumed 3-genre classification
	model.add(Activation('softmax'))
	if optimiser in [None, 'sgd']:
		print ":::use sgd optimiser"
		sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=sgd)
	else:
		print ":::use rmsprop optimiser"
		rmsprop = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
		model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

	return model

def load_weights():
	''' Load keras config file and return W
	'''
	model_name = "vggnet5"
	keras_filename = "vggnet5_local_keras_model_CNN_stft_11_frame_173_freq_257_folding_0_best.keras"
	
	num_fr = 173
	len_freq = 257
	learning_rate=0.0005
	dropout_values = [0, 0.25, 0.25, 0.25, 0.5]
	print '--- load model ---'
	
	model = buildConvNetModel(num_fr, len_freq, learning_rate, model_name, dropout_values, 'rmsprop')
	model.load_weights(keras_filename)
	# understand the current model
	W = []
	layer_names = []
	for layer_ind, layer in enumerate(model.layers):
		layer_config_here = layer.get_config()
		print '%d-th layer with name: %s' % (layer_ind, layer_config_here['name'])
		layer_names.append(layer_config_here['name'])
		if layer_config_here['name'] == 'Convolution2D':
			W.append(layer.W.get_value(borrow=True))

	return W


if __name__ == "__main__":
	'''
	This is main body of program I used for the paper at ismir. 
	You need a keras model weights file with music files at the appropriate folder... In other words, it won't be executed at your computer.
	Just see the part after 
		print '--- deconve! ---'
	, where the deconvolution functions are used.

	'''
	W = load_weights()
	num_conv_layer = len(W)
	
	# load files
	print '--- prepare files ---'
	filenames_src = ['42.mp3', '3695701.mp3', '3696.mp3', '2489888.mp3', 'Chopin.m4a', 'NIN.m4a', 'eminem.mp3', 'neverendingstory.mp3', 'toy.MP3', 'Babybaby.mp3', 'michel.m4a', 'bach.m4a', 'beethoven.m4a', 'Dream.mp3'] # 42 김건모 미련, 3695701 버벌진트 시작이좋아, 3696 김사랑 feeling, 2489888 (락) Time-Bomb 

	N_FFT = 512
	SAMPLE_RATE = 11025
	path_SRC = '/Users/naver/Documents/keunwoo_src/'
	path_src = '/Users/naver/Documents/keunwoo_src/'

	# get STFT of the files.
	for filename in filenames_src:
		song_id = filename.split('.')[0]
		if os.path.exists(path_SRC + song_id + '.npy'):
			pass
		else:
			src = librosa.load( os.path.join(path_src, filename), sr=SAMPLE_RATE, mono=True, offset=30., duration=4.)[0]
			SRC = librosa.stft(src, n_fft=N_FFT, hop_length=N_FFT/2)
			np.save(path_SRC + song_id + '.npy', SRC)

	# deconve
	depths = [5,4,3,2,1]
	#for filename in ['42.npy', '995-0.npy']:
	filenames_SRC = ['eminem.npy', 'neverendingstory.npy', 'toy.npy', 'Babybaby.npy', 'michel.npy', 'bach.npy', 'beethoven.npy', 'Dream.npy', 'Chopin.npy', 'NIN.npy'] # ['42.npy', '3696.npy', '2489888.npy', '3695701.npy']
	for filename in filenames_SRC:
		#filename = filenames_SRC[0]
		for depth in depths:
			song_id = filename.split('.')[0]
		
			# song_id = filename.split('.')[0]
			# SRC =np.load(path_SRC + song_id + '.npy')
			SRC = np.load(path_SRC + filename)
			filename_out = '%s_a_original.wav' % (song_id)	
			if not os.path.exists(path_SRC + song_id):
					os.makedirs(path_src + song_id)
			if not os.path.exists(path_SRC + song_id + '_img'):
					os.makedirs(path_src + song_id + '_img')
			
			librosa.output.write_wav(path_src+ song_id + '/' + filename_out, librosa.istft(SRC, hop_length=N_FFT/2), sr=SAMPLE_RATE, norm=True)
			
			print '--- deconve! ---'
			# if os.path.exists(path_SRC + song_id + '_deconvedMASKS.npy'):
			# 	deconvedMASKS = np.load(path_SRC + song_id + '_deconvedMASKS.npy')
			# else:
			deconvedMASKS = auralise.get_deconve_mask(W[:depth], layer_names, SRC, depth) # size can be smaller than SRC due to downsampling
			# np.save(path_SRC + song_id + '_deconvedMASKS.npy', deconvedMASKS)
			
			print 'result; %d masks with size of %d, %d' % deconvedMASKS.shape
			for deconved_feature_ind, deconvedMASK_here in enumerate(deconvedMASKS):
				MASK = np.zeros(SRC.shape)
				MASK[0:deconvedMASK_here.shape[0], 0:deconvedMASK_here.shape[1]] = deconvedMASK_here
				deconvedSRC = np.multiply(SRC, MASK)

				
				filename_out = '%s_deconved_from_depth_%d_feature_%d.wav' % (song_id, depth, deconved_feature_ind)
				librosa.output.write_wav(path_src+ song_id + '/' + filename_out, librosa.istft(deconvedSRC, hop_length=N_FFT/2), SAMPLE_RATE, norm=True)
				filename_img_out = 'spectrogram_%s_from_depth_%d_feature_%d.png' % (song_id, depth, deconved_feature_ind)
				imsave(path_src+song_id + '_img' + '/' + filename_img_out , np.flipud(np.multiply(np.abs(SRC), MASK)))

				filename_img_out = 'filter_for_%s_from_depth_%d_feature_%d.png' % (song_id, depth, deconved_feature_ind)
				imsave(path_src+song_id + '_img' + '/' + filename_img_out , np.flipud(MASK))
				
