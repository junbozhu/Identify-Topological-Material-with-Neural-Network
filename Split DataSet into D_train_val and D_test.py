'''
This py load whole data set (after feature representation process),
Then divide dataset into D_train_val and D_test, and save them into txt file
D_train_val is used for cross-validation
D_test is reserved for final test
'''

import numpy as np
import os

# Loading dataset:
data = np.loadtxt("data_repre2.txt", unpack=False)
N,D = data.shape # N = 28605 number of data points, D=145 feature dimension

label1, label2 = np.loadtxt("labels.txt", unpack=True)
labels_binary_soc = np.array([label1]).T # note: label1 is calculated by DFT with SOC
label = labels_binary_soc # (N,1), 1 -> non-trivial class, 0 -> trivial class
# we use nn.CrossEntropyLoss(). label is just an integer (not one-hot)
print(f'{np.sum(label)/N*100:.1f}% of all data are class 1 (topo non-trivial)')

''' #if not using nn.CrossEntropyLoss(), one-hot the label
labels_binary_soc = np.array([label1]).T # note: label1 is calculated by DFT with SOC
label = np.zeros([N,2]) # prepare for softmax
label[:,0:1] = labels_binary_soc # first row means topo-nontrivial
label[:,1:2] = 1 - labels_binary_soc # second row means topo-trivial
'''

# pre-shuffling of data
np.random.seed(17) # set random_seeds for reproducibility, let me use 17
Temp = np.concatenate((data,label),1)
np.random.shuffle(Temp) # shuffled along axis 0
data = Temp[:,0:D]
label = Temp[:,D:] # (N,1)
del Temp # free memory
label = label[:,0] # get (N,)
# reduce array dimension for for nn.CrossEntropyLoss(). label should be an integer

print('Data loaded!')
print("Confirm data shape: " + str(data.shape))
print("Confirm label shape: " + str(label.shape))

# split out 10% data as test data set:
test_ratio = 0.1
test_size = int(N*test_ratio)
data_test, data_train_val = np.split(data,[test_size]) # note: we shuffled data already,
label_test, label_train_val = np.split(label,[test_size])
print(f'{np.sum(label_test)/len(label_test)*100:.1f}% of test data are class 1 (topo non-trivial)')
print(f'{np.sum(label_train_val)/len(label_train_val)*100:.1f}% of train&val data are class 1 (topo non-trivial)')

'''

# save Data_train_val and Data_test to data file
codeDir = os.path.dirname(os.path.realpath('__file__'))
fileDir = os.path.join(codeDir,'data_repre2_train_val_test')

if not os.path.exists(fileDir):
    os.makedirs(fileDir)
    
with open(os.path.join(fileDir,'data_train_val.txt'),'w') as f: # (N_train_val, D)
        np.savetxt(f, data_train_val,fmt='%.8f',header="data_train_val")
        
with open(os.path.join(fileDir,'label_train_val.txt'),'w') as f: # (N_train_val, )
        np.savetxt(f, label_train_val,fmt='%.8f',header="label_train_val")
        
with open(os.path.join(fileDir,'data_test.txt'),'w') as f: # (N_test, D)
        np.savetxt(f, data_test,fmt='%.8f',header="data_test")
        
with open(os.path.join(fileDir,'label_test.txt'),'w') as f: # (N_test, )
        np.savetxt(f, label_test,fmt='%.8f',header="label_test")

     '''   
