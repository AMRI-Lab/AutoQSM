import numpy as np
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from util import *
from model import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95 
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
KTF.set_session(sess)
tf.global_variables_initializer().run(session=sess)

def train(	model_name, 
			output_metrics_name,
			data_read_path,
			input_patch_shape,
			output_patch_shape,
			batch_size,
			learning_rate,
			epochs,
			output_symbol
		):

	model_input_path  = '../models/{}'.format(output_metrics_name)


	train_dataread = data_read(np.arange(1,14), pathname=data_read_path)
	print('1')
	valid_dataread = data_read(np.arange(15,17), pathname=data_read_path)
	print('2')
	train_datagenerate = data_generate(train_dataread, batch_size=batch_size, 
		input_data_patch_shape=input_patch_shape, 
		output_data_patch_shape=output_patch_shape,
		output_symbol = output_symbol)
	valid_datagenerate = data_generate(valid_dataread, batch_size=batch_size, input_data_patch_shape=input_patch_shape, output_data_patch_shape=output_patch_shape,output_symbol = output_symbol)
	
	model = model_name(input_patch_shape, output_patch_shape)
	model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

	model_check_point = ModelCheckpoint(filepath='{}/model_{}.hdf5'.format(model_input_path, '1'), 
								verbose=1, 
								monitor='val_loss',
								save_best_only=True)
	
	tensor_board = TensorBoard(log_dir='/home/zhangm/upload/new/logs/{}/{}'.format(output_metrics_name, '1'))
	early_stop = EarlyStopping(monitor='val_loss', patience=50)
	print('ok')						
	model.fit_generator(train_datagenerate, 
						steps_per_epoch  = 200,
						epochs			 = epochs,
						validation_data  = valid_datagenerate,
						validation_steps = 100,
						callbacks		 = [model_check_point, tensor_board, early_stop],
						verbose			 = 1)
	model.save('{}/model_final_{}.hdf5'.format(model_input_path, '1'))


if __name__ == '__main__':
	train(	model_name			= vnet, 
			output_metrics_name	='vnet', 
			data_read_path 		= '../train_data/',
			input_patch_shape   = [64, 64, 64],
			output_patch_shape  = [32, 32, 32],
			batch_size 			= 8,
			learning_rate 		= 1e-4,
			epochs				= 500,
			output_symbol		= ['qsm_images'])
