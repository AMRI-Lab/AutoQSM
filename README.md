# AutoQSM  
Source codes and trained networks described in the paper: Learning-based single-step quantitative susceptibility mapping reconstruction without brain extraction

###Environmental Requirements:  

Python 3.6  
Tensorflow 1.15.0  
Keras 2.2.5  

###Files descriptions:  
AutoQSM contains the following folders:  

code: It contains the source codes for training and testing.  

model/vnet/model_final_1.hdf5: Weights of AutoQSM


###Usage  
##Test  
You can run code/test.py to test the network performance on the provided data. The results will be in the 'results' directory    
For test on your own data. You can use "data_predict" function as shown in test_demo.py files  

##train  
If you want to train AutoQSM by yourself. code/train.py can be used as a reference.  


