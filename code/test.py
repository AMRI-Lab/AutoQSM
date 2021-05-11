import os
from scipy.io import savemat
from model import *
from util import *

import numpy as np
from scipy.io import loadmat
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def test(model_name, 
		 model_path, 
		 unwrap_data_patch_shape, 
		 output_patch_shape,
		 test_data_order, 
		 test_data_path, 
		 output_data_path,
        symbol):
    
    model = model_name(unwrap_data_patch_shape, output_patch_shape, 1)
    model.load_weights(model_path)
    
    test_data = data_read(test_data_order, test_data_path)
    
    
    for j, data_symbol in enumerate(test_data):
        print('this is for',j)
        unwrap_data = data_symbol[symbol]
        
        data_symbol['autoQSM_images'], data_symbol['patch_images'] = data_predict(model, unwrap_data, unwrap_data_patch_shape, output_patch_shape)
        savemat('{}/subject_{}.mat'.format(output_data_path, test_data_order[j]), data_symbol)


if __name__ == '__main__':
	test(
		model_name			= vnet, 
		model_path  		= '../models/vnet/model_final_1.hdf5', 
		unwrap_data_patch_shape   = [64, 64, 64], 
		output_patch_shape  = [32,32,32],
		test_data_order 	= np.arange(0,1), 
		test_data_path 		= '../test_data/',
		output_data_path 	= '../results/',
        symbol='x_input')
