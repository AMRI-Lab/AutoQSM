# AutoQSM  
Source codes and trained networks described in the paper: Learning-based single-step quantitative susceptibility mapping reconstruction without brain extraction

###Environmental Requirements:  

Python 3.6  
Tensorflow 1.15.0  
Keras 2.2.5  

###Files descriptions:  
AutoQSM contains the following folders:  

test_data: It provides three test data.

logs/last.h5: A file that contains the weights of the trained model

model/MoDL_QSM.py : This file contains the functions to create the model-based convolutional neural network proposed in our paper

test: It contains test_tools.py and test_demo.py. test_tools.py offers some supporting functions for network testing such as image patch stitching, dipole kernel generation, etc. test_demo.py shows how to perform network testing with data from the "data" folder

train: It contains train_lines.py. train_gen.py: This is the code for network training

NormFactor.mat: The mean and standard deviation of our training dataset for input normalization.

###Usage
##Test
You can run test_demo.py directly to test the network performance on the provided data. The results will be in the same directory as the input data
For test on your own data. You can use "model_test" function as shown in test_demo.py files

##train
If you want to train MoDL-QSM by yourself. train_lines.py can be used as a reference.


