# AutoQSM  
Source codes and trained networks described in the paper: Learning-based single-step quantitative susceptibility mapping reconstruction without brain extraction  
Link: https://www.sciencedirect.com/science/article/pii/S1053811919306469  

###Environmental Requirements:  

Python 3.6  
Tensorflow 1.15.0  
Keras 2.2.5  

###Files descriptions:  
AutoQSM contains the following folders:  

code: It contains the source codes for training and testing.  

models/vnet/model_final_1.hdf5: Weights of AutoQSM.  

test_data: one data for testing.   


###Usage  
##Test  
You can run code/test.py to test the network performance on the provided data. The output will be in the 'results' directory    
For test on your own data. You can use "data_predict" function as shown in test.py files  

##train  
If you want to train AutoQSM by yourself. code/train.py can be used as a reference.  


