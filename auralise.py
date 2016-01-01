
import numpy as np
from scipy import signal
import scipy.ndimage

def get_deconve_mask(W, layer_names, SRC, depth):
	''' 
	This function returns the deconvolved mask.

	W: weights. shape: (#conv_layers -by- #channels_out - #channels_in - #rows - #cols)
	layer_names: array of layer names from keras
	SRC: STFT representation 
	depth : integer, 0,1,2,3,.. if it's 5, and if there were 5 conv-MP layers, it will deconve from the deepest (i.e. the highest-level feature)

	'''
	def relu(x):
		return np.maximum(0., x)

	def get_deconvolve(images, weights):
		''' input image is expected to be unpooled image, i.e. 'upsampled' with switch matrix. 
		weights: 4-d array. (#channels_out - #channels_in - #rows - #cols) e.g. 64-64-3-3. 
		         When deconvolve, 'out' is input of deconvolution and vice versa. 
		         Check the names below: 'num_before_deconv' and 'num_after_deconv', which are self-explanatory 
		'''
		num_before_deconv, num_after_deconv, num_row, num_col = weights.shape
		flipped_weights = weights[:, :, ::-1, ::-1] # fliplr and flipud to use in deconvolution.
		reversed_flipped_weights = np.zeros((num_after_deconv, num_before_deconv, num_row, num_col))
		for dim0 in xrange(num_after_deconv): # reverse the dimension to reuse get_convolve function.
			for dim1 in xrange(num_before_deconv):
				reversed_flipped_weights[dim0, dim1, :, :] = flipped_weights[dim1, dim0, :, :]
		
		return get_convolve(images, reversed_flipped_weights)

	def get_unpooling2d(images, switches, ds=2):
		'''input imge size is (almost) half of switch'''
		num_image, num_img_row, num_img_col = images.shape
		num_switch, num_swt_row, num_swt_col = switches.shape
		# out_images = np.zeros((num_image, num_row*ds, num_col*ds))
		out_images = np.zeros((num_image, num_swt_row, num_swt_col))
		for ind_image, image in enumerate(images):
			out_images[ind_image, :num_img_row*ds, :num_img_col*ds] = np.multiply(scipy.ndimage.zoom(image, ds, order=0), switches[ind_image, :num_img_row*ds, :num_img_col*ds]) # [1 ] becomes [1 1; 1 1], then multiplied.
		return out_images

	def get_convolve(images, weights):
		''' images: 3-d array, #channel-#rows-#cols e.g. (1,257,173)
			weights: 4-d array, (#channels_out - #channels_in - #rows - #cols) e.g. (64,1,3,3) for the first convolution
		'''
		num_out, num_in, num_row_w, num_col_w = weights.shape
		num_row_img, num_col_img = images.shape[1], images.shape[2]
		out_images = np.zeros((num_out, num_row_img, num_col_img))
		
		for ind_input_layer in xrange( weights.shape[1] ):
			for ind_output_layer in xrange( weights.shape[0] ):
				
				out_images[ind_output_layer, :, :] += signal.convolve2d(images[ind_input_layer, :, :], weights[ind_output_layer, ind_input_layer, :, :], mode='same')

		return out_images

	def get_MP2d(images, ds=2):
		''' 
		images: 3-d array, #channel-#rows-#cols e.g. (1,257,173)
		ds = integer, which downsample by. e.g. 2
		return: result and switch
		    result: downsampled images (with 2d MAX)
		    switch: list of switch matrix; that showing from which position each 'max' values come.

		* ignore_border = True is assumed.
		'''
		
		num_image, num_row, num_col = images.shape
		out_images = np.zeros((num_image, num_row/ds, num_col/ds))
		switch = np.zeros((num_image, num_row, num_col))
		
		for ind_image, image in enumerate(images):
			for row_ind in xrange(num_row/ds):
				for col_ind in xrange(num_col/ds):
					out_images[ind_image, row_ind, col_ind] = np.max( image[ds*row_ind:ds*row_ind+ds, ds*col_ind:ds*col_ind+ds] )
					argmax_here = 						   np.argmax( image[ds*row_ind:ds*row_ind+ds, ds*col_ind:ds*col_ind+ds] )
					switch[ind_image, ds*row_ind+argmax_here/ds, ds*col_ind+argmax_here%ds] = 1

		return out_images, switch

	'''function body of get_deconve_mask begins here.'''
	MAG = []
	MAG.append( np.zeros((1, SRC.shape[0], SRC.shape[1])) )
	
	MAG[0][0,:,:] = np.abs((SRC))
	
	switch_matrices = []
	procedures = []
	conv_ind = 0
	mp_ind = 0
	
	# [1] feed-forward path.
	print '-------feed-forward-'
	for layer_ind, layer_name in enumerate(layer_names):
		if layer_name == "Convolution2D":
			MAG.append(relu(get_convolve(images=MAG[-1], weights=W[conv_ind])))
			procedures.append('conv')
			conv_ind += 1

		elif layer_name == "MaxPooling2D":
			result, switch = get_MP2d(images=MAG[-1], ds=2)
			MAG.append(result)
			procedures.append('MP')
			switch_matrices.append(switch)
			mp_ind += 1

		if mp_ind == depth:
			break;

		elif layer_name == "Flatten":
			break
	
	# [2] 'deconve' # numbers below come from vggnet5 model (when depth == 4).
	revMAG = list(reversed(MAG)) # len(revMAG)==9, revMAG[i].shape = (64,16,10), (64,32,21), (64,32,21), (64,64,43), (64,128,86),(64,128,86),(64,257,173),(1,257,173)
	revProc = list(reversed(procedures)) # len(revProc)==8, ['MP', 'conv', 'MP', 'conv', 'MP', 'conv', 'MP', 'conv']
	revSwitch = list(reversed(switch_matrices)) # len(revSwitch)==4, revSwitch[0].shape = (64, 32, 21), (64, 64, 43), (64, 128, 86), (64, 257, 173)
	revW = list(reversed(W)) #len(revW)==4, (64,64,3,3), (64,64,3,3), (64,64,3,3), (64,1,3,3)

	num_outputs = revMAG[0].shape[0] # number of channels in the layer we consider (layer at 'depth')
	
	deconved_final_results = np.zeros((num_outputs, SRC.shape[0], SRC.shape[1]))
	
	for ind_out in xrange(num_outputs): 
		# Init with values that only (ind_out)-th feature map be considered
		deconvMAG 				= [None]
		deconvMAG[0] 			= np.zeros((1, revMAG[0].shape[1], revMAG[0].shape[2]))
		deconvMAG[0][0, :, :] 	= revMAG[0][ind_out, :, :] # assign with the spectrogram at the Last stage (at 'depth'). 

		revSwitch_to_use 				= [None]*len(revSwitch)
		revSwitch_to_use[0] 			= np.zeros((1, revSwitch[0].shape[1], revSwitch[0].shape[2]))
		revSwitch_to_use[0][0, :, :] 	= revSwitch[0][ind_out, :, :]
		revSwitch_to_use[1:]			= revSwitch[1:]

		revW_to_use    		= [None] * len(revW)
		revW_to_use[0]		= np.zeros((1, revW[0].shape[1], revW[0].shape[2], revW[0].shape[3]))
		revW_to_use[0][0,:,:,:] = revW[0][ind_out, :, :, :] #only weights to yield (ind_out)-th feature map.
		revW_to_use[1:]			= revW[1:]

		# Go!
		print '-------feed-back- %d --' % ind_out
		unpool_ind = 0
		deconv_ind = 0
		for proc_ind, proc_name in enumerate(revProc):
			if proc_name == 'MP':
				# print 'unpool: %d, %d' % (unpool_ind, proc_ind)
				deconvMAG.append(relu(get_unpooling2d(images=deconvMAG[proc_ind], switches=revSwitch_to_use[unpool_ind])))
				unpool_ind += 1

			elif proc_name == "conv":
				# print 'deconv: %d, %d' % (deconv_ind, proc_ind)
				deconvMAG.append(get_deconvolve(images=deconvMAG[proc_ind], weights=revW_to_use[deconv_ind]))
				deconv_ind += 1

		deconved_final_results[ind_out, :, :] = deconvMAG[-1][:,:,:]
	
	return deconved_final_results

def load_weights():
	''' Load keras config file and return W
	'''
	import h5py
	model_name = "vggnet5"
	keras_filename = "vggnet5_local_keras_model_CNN_stft_11_frame_173_freq_257_folding_0_best.keras"
	
	print '--- load model ---'
	
	W = []
	f = h5py.File(keras_filename)
	for idx in xrange(f.attrs['nb_layers']):
		key = 'layer_%d' % idx
		if f[key].keys() != []:
			W.append(f[key]['param_0'][:,:,:,:])
		if len(W) == 5:
			break
	layer_names = []
	for idx in xrange(5):
		layer_names.append('Convolution2D')
		layer_names.append('MaxPooling2D')
	layer_names.append('Flatten')
	
	return W, layer_names

