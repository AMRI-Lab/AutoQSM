import numpy as np
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D , Conv3DTranspose, Cropping3D , concatenate

def vnet(input_shape=[64, 64, 16], output_shape=[32, 32, 8], n=1):

	input_layer = Input(shape=input_shape+[n])

	layer_conv0 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(input_layer)
	layer_conv0 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(layer_conv0)
	layer_pool0 = MaxPooling3D(pool_size=(2, 2, 2))(layer_conv0)

	layer_conv1 = Conv3D(16*2, (3, 3, 3), activation='relu', padding='same')(layer_pool0)
	layer_conv1 = Conv3D(16*2, (3, 3, 3), activation='relu', padding='same')(layer_conv1)
	layer_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(layer_conv1)

	layer_conv2 = Conv3D(16*4, (3, 3, 3), activation='relu', padding='same')(layer_pool1)
	layer_conv2 = Conv3D(16*4, (3, 3, 3), activation='relu', padding='same')(layer_conv2)
	layer_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(layer_conv2)

	layer_conv3 = Conv3D(16*8, (3, 3, 3), activation='relu', padding='same')(layer_pool2)
	layer_conv3 = Conv3D(16*8, (3, 3, 3), activation='relu', padding='same')(layer_conv3)
	
	layer_up4 = Conv3DTranspose(16*4, (2, 2, 2), strides=(2, 2, 2), padding='same')(layer_conv3)
	layer_up4 = concatenate([layer_up4, layer_conv2], axis=4)
	layer_conv4 = Conv3D(16*4, (3, 3, 3), activation='relu', padding='same')(layer_up4)
	layer_conv4 = Conv3D(16*4, (3, 3, 3), activation='relu', padding='same')(layer_conv4)

	layer_up5 = concatenate([Conv3DTranspose(16*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(layer_conv4), layer_conv1], axis=4)
	layer_conv5 = Conv3D(16*2, (3, 3, 3), activation='relu', padding='same')(layer_up5)
	layer_conv5 = Conv3D(16*2, (3, 3, 3), activation='relu', padding='same')(layer_conv5)

	layer_up6 = concatenate([Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(layer_conv5), layer_conv0], axis=4)
	layer_conv6 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(layer_up6)
	layer_conv6 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(layer_conv6)

	layer_conv7 = Conv3D(1, (1, 1, 1), activation='tanh')(layer_conv6)

	tempx, tempy, tempz = np.round((np.array(input_shape) - np.array(output_shape))/2).astype(int)
	decoded  = Cropping3D(((tempx, tempx), (tempy, tempy), (tempz, tempz)))(layer_conv7)

	predictions  = Model(input_layer, decoded)

	return predictions 

	
if __name__ == '__main__':
	model = vnet()
	print(model.summary())
