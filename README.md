# AutoQSM  
Source codes and trained networks described in the paper: Learning-based single-step quantitative susceptibility mapping reconstruction without brain extraction

###Environmental Requirements:  

Python 3.6  
Tensorflow 1.15.0  
Keras 2.2.5  

###Files descriptions:  
AutoQSM contains the following folders:  

code: It contains the source codes for training and testing.  

test_data: It provides three test data.  

logs/last.h5: A file that contains the weights of the trained model

model/MoDL_QSM.py : This file contains the functions to create the model-based convolutional neural network proposed in our paper


###Usage
##Test
You can run code/test.py directly to test the network performance on the provided data. The results will be in the 'results' directory  
For test on your own data. You can use "data_predict" function as shown in test_demo.py files

##train
If you want to train AutoQSM by yourself. code/train.py can be used as a reference.


