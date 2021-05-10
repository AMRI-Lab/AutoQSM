import numpy as np
from scipy.io import loadmat
import random
#from math import *
def data_read(data_orders, pathname):
	read_data = []
	for i in data_orders:
		data_temp = loadmat(pathname + str(i) + '.mat')
		read_data.append(data_temp)
	return read_data


def data_generate(data, batch_size, input_data_patch_shape, output_data_patch_shape, output_symbol):
    while True:
        batch_X = []
        batch_Y = []
        for i in range(batch_size):
            temp = np.random.choice(len(data))
            symbol = random.choice(output_symbol)
            image_shape = data[temp]['unwrap_images'].shape
            
            random_x = np.random.choice(np.arange(int(input_data_patch_shape[0]/2), image_shape[0]-int(input_data_patch_shape[0]/2)))
            random_y = np.random.choice(np.arange(int(input_data_patch_shape[1]/2), image_shape[1]-int(input_data_patch_shape[1]/2)))
            random_z = np.random.choice(np.arange(int(input_data_patch_shape[2]/2), image_shape[2]-int(input_data_patch_shape[2]/2)))
            
            phase_image = data[temp]['unwrap_images'][random_x-int(input_data_patch_shape[0]/2):random_x+int(input_data_patch_shape[0]/2), 
				random_y-int(input_data_patch_shape[1]/2):random_y+int(input_data_patch_shape[1]/2), random_z-int(input_data_patch_shape[2]/2):random_z+int(input_data_patch_shape[2]/2)]
            
            magnitude_image = data[temp]['magnitude_images'][random_x-int(input_data_patch_shape[0]/2):random_x+int(input_data_patch_shape[0]/2), 
				random_y-int(input_data_patch_shape[1]/2):random_y+int(input_data_patch_shape[1]/2), random_z-int(input_data_patch_shape[2]/2):random_z+int(input_data_patch_shape[2]/2)]
            
            qsm_image = data[temp][symbol][random_x-int(output_data_patch_shape[0]/2):random_x+int(output_data_patch_shape[0]/2), 
				random_y-int(output_data_patch_shape[1]/2):random_y+int(output_data_patch_shape[1]/2), random_z-int(output_data_patch_shape[2]/2):random_z+int(output_data_patch_shape[2]/2)]
            
            batch_X_temp = np.stack([phase_image, magnitude_image/32767.], axis=-1)
            batch_Y_temp = np.expand_dims(qsm_image, axis=3)
            
            batch_X.append(batch_X_temp)
            batch_Y.append(batch_Y_temp)
        yield np.array(batch_X)[:,:,:,:,[0]], np.array(batch_Y)


def data_predict(model_name, input_data, input_data_patch_shape, output_data_patch_shape):
    shift=24;
    temp_X, temp_Y, temp_Z = input_data.shape
    temp_x, temp_y, temp_z = output_data_patch_shape
    temp_xi, temp_yi, temp_zi = input_data_patch_shape
    temp_px, temp_py, temp_pz = int((temp_xi-temp_x)/2), int((temp_yi-temp_x)/2), int((temp_zi-temp_z)/2) 
    temp_pad_x, temp_pad_y, temp_pad_z = int(np.ceil((temp_X-temp_x)/shift)*shift-(temp_X-temp_x)), int(np.ceil((temp_Y-temp_y)/shift)*shift-(temp_Y-temp_y)), int(np.ceil((temp_Z-temp_z)/shift)*shift-(temp_Z-temp_z))
    input_data_pad = np.pad(input_data, ((0, temp_pad_x), (0, temp_pad_y), (0, temp_pad_z)), 'edge')
    
    
    temp_Xp, temp_Yp, temp_Zp = input_data_pad.shape
    output = np.zeros_like(input_data_pad)
    input_data_pad = np.pad(input_data_pad, ((temp_px, temp_px), (temp_py, temp_py), (temp_pz, temp_pz)), 'edge')
    
    output_patches = []
    num_k=int((temp_Zp-temp_z)/shift+1)
    num_j=int((temp_Yp-temp_y)/shift+1)
    num_i=int((temp_Xp-temp_x)/shift+1)
    for k in range(num_k):
        for j in range(num_j):
            for i in range(num_i):
                input_data_patch = input_data_pad[shift*i:(shift*i+temp_xi), shift*j:(shift*j+temp_yi), shift*k:(shift*k+temp_zi)]
                input_data_patch = np.expand_dims(input_data_patch, axis=0)
                input_data_patch = np.expand_dims(input_data_patch, axis=-1)
                output_patch = model_name.predict(input_data_patch)
                temp_patch=output_patch[0,:,:,:,0]
             #   output[temp_x*i:(temp_x*i+temp_x), temp_y*j:(temp_y*j+temp_y), temp_z*k:(temp_z*k+temp_z)] = output_patch[0,:,:,:,0]                              
                if i!=0:
                    patch2=output_patches[-1]
                    temp_patch=patch_process(temp_patch,patch2,8,0)
                if j!=0:
                    patch2=output_patches[-num_i]
                    temp_patch=patch_process(temp_patch,patch2,8,1)
                if k!=0:
                    patch2=output_patches[-num_i*num_j]
                    temp_patch=patch_process(temp_patch,patch2,8,2)
                    
                output[shift*i:(shift*i+temp_x),shift*j:(shift*j+temp_y),shift*k:(shift*k+temp_z)]=temp_patch
                #output_patches.append(output_patch[0,:,:,:,0])
                output_patches.append(temp_patch)
                
    outputs_data = output[:temp_X, :temp_Y, :temp_Z]
    output_patches_data = np.array(output_patches)
    return outputs_data, output_patches_data



def patch_process(patch1,patch2,overlap,direction):
    #image patches splicing
    #patch1: patch to be spliced
    #patch2: the patch overlapped with patch1
    #direction: The direction of splicing    
    
    size1=patch1.shape
    size2=patch2.shape
    result=patch1
    if size1==size2:
        size=size1
    else:
        raise ValueError("two patches don't have the same size")

             
    weight=np.ones(overlap) 
    for x in range(int(overlap)):
        weight[x]=1-x/(overlap-1)
            
            
    if direction==0:
        block1=patch1[0:overlap,:,:]
        block2=patch2[int(size[0]-overlap):size[0],:,:]
        for i in range(int(overlap)):
            result[i,:,:]=weight[i]*block2[i,:,:]+(1-weight[i])*block1[i,:,:]
    
    elif direction==1:
        block1=patch1[:,0:overlap,:]
        block2=patch2[:,int(size[1]-overlap):size[1],:]
        for j in range(int(overlap)):
            result[:,j,:]=weight[j]*block2[:,j,:]+(1-weight[j])*block1[:,j,:]
            
    elif direction==2:
        block1=patch1[:,:,0:overlap]
        block2=patch2[:,:,int(size[2]-overlap):size[2]]
        for k in range(int(overlap)):
            result[:,:,k]=weight[k]*block2[:,:,k]+(1-weight[k])*block1[:,:,k]
    else:
        raise ValueError("direction must be a integer between 0 and 2")
            
    return result
        
            
        
    
    


