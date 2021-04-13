import torch, torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


#set random seeds for reproducibility, let's use 17
torch.manual_seed(17)
torch.cuda.manual_seed(17)
np.random.seed(17)

'''Device configuration'''
device = torch.device('cpu') 
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Working on device: {device}')


'''Loading dataset:'''
dataDir = 'C:\\Users\\Junbo Zhu\\Documents\\6.036 intro to ML\\2021 spring 6-862\\project code\\data_repre2_train_val_test'
data_train_val = np.loadtxt(os.path.join(dataDir,'data_train_val.txt'), unpack=False)  # (N_train_val, D)
label_train_val = np.loadtxt(os.path.join(dataDir,'label_train_val.txt'), unpack=False) # (N_train_val, )
#data_test = np.loadtxt(os.path.join(dataDir,'data_test.txt'), unpack=False) # (N_test, D)
#label_test = np.loadtxt(os.path.join(dataDir,'label_test.txt'), unpack=False) # (N_test, )
N_train_val, D = data_train_val.shape


'''set mini-batchsize and epochs'''
batchsize = 64
print(f'minibatch size = {batchsize}')
epochs = 32 # one epoch is a loop of whole train_dataset
print(f'epochs = {epochs}')


'''set model, loss function, optimizer:'''

Linear_model = torch.nn.Sequential(
    torch.nn.Linear(D, 2, bias=True), #input dim is feature dim = D = 145
    # Note I use CrossEntropyLoss (logSoftMax + Nll) so here no need Softmax layer
)


units = [80,20] # hidden layer units number
'''
nn_onelayer_Relu = torch.nn.Sequential(
    torch.nn.Linear(D, units[0], bias=True), #input dim is feature dim = D = 145
    torch.nn.ReLU(),
    torch.nn.Linear(units[0], 2, bias=True),
    # Note I use CrossEntropyLoss (logSoftMax + Nll) so here no need Softmax layer
)'''

nn_twolayer_Relu = torch.nn.Sequential(
    torch.nn.Linear(D, units[0], bias=True), #input dim is feature dim = D = 145
    torch.nn.ReLU(),
    torch.nn.Linear(units[0], units[1], bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(units[1], 2, bias=True),
    # Note I use CrossEntropyLoss (logSoftMax + Nll) so here no need Softmax layer
)

'''
nn_threelayer_Relu = torch.nn.Sequential(
    torch.nn.Linear(D, units[0], bias=True), #input dim is feature dim = D = 145
    torch.nn.ReLU(),
    torch.nn.Linear(units[0], units[1], bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(units[1], units[2], bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(units[2], 2, bias=True),
    # Note I use CrossEntropyLoss (logSoftMax + Nll) so here no need Softmax layer
)'''

model = nn_twolayer_Relu.double() # both model and data use double precision
model.to(device)
criterion = nn.CrossEntropyLoss() # loss function: CrossEntropyLoss, default to get averaged loss
# use Adam, with default lr, betas etc
lrate = 1e-3 # default 1e-3
betas = (0.9, 0.999) # default (0.9, 0.999)
eps = 1e-08 # default 1e-08
decay = 0 # default 0
amsgrad = False # default False
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

'''define dataclass etc:'''

def init_normal(m): # initialization function for neural network
    if type(m) == nn.Linear: # check the module type
        nl = m.in_features
        m.weight.data.normal_(0.0,1/np.sqrt(nl)) # mean=0, std = 1/sqrt(nl)
        m.bias.data.fill_(0) # set bias 0

# use the modules apply function to recursively apply the initialization
model.apply(init_normal)

# Create a Dataset class to use with PyTorch's Built-In Dataloaders
class topoDataset(Dataset):
    def __init__(self, data, label):
        self.data =  data # (N,D) array, D is feature dim, N is number of data points
        self.label = label # (N,) array

    def __len__(self): #returns N i.e. number of data points
        return self.data.shape[0] #note data.shape is NxD
        #note: call by len(XXX)

    def  __getitem__(self, ind): #returns a dataset element given an index
        return (self.data[ind,:], self.label[ind] )
        #note: call by XXX[ind], ind can be like 1:10 etc

    def split_val(self, n_start, n_end): #n_start,n_end are index range of val data, prepare for cross-validation
        data_val, label_val = self[n_start:n_end]
        D_val = topoDataset(data_val,label_val)
        data_train = np.concatenate((self[:n_start][0],self[n_end:][0]),0)
        label_train = np.concatenate((self[:n_start][1],self[n_end:][1]),0)
        D_train = topoDataset(data_train,label_train)
        return (D_train,D_val)


def train(model, device, train_loader, criterion, optimizer):
    '''
    Function for training the networks. One call to train() performs a single
    epoch for training.
    '''
    # Set the model to training mode.
    model.train()

    #we'll keep adding the loss of each minibatch to total_loss, so we can calculate
    #the average loss at the end of the epoch.
    total_loss = 0
    total_correct = 0
    total_items = 0
    '''
    # Iterate through the batch (whole train dataset) means 1 epoch.
    # One call of train() trains for 1 epoch.
    # batch_idx: an integer representing which batch number we're on
    # update model with minibatch
    # input: a pytorch tensor representing a batch of input.'''
    for batch_idx, (input,target) in enumerate(train_loader):
        # This line sends data to GPU if you're using a GPU
        input = input.double()# both model and data use double precision
        input = input.to(device)
        target = target.type(torch.LongTensor).to(device)
        #print(input.shape)
        
        #Pytorch accumulates gradients, best practice to zero all of the gradients currently tracked by the optimizer 
        #before performing the next parameter up-date
        optimizer.zero_grad()

        # feed our input through the network
        output = model.forward(input)

        # calculate Loss
        loss_value = criterion( output, target )

        # Perform backprop
        loss_value.backward()
        optimizer.step()

        #accumulate loss to later calculate the average
        total_loss += loss_value
        total_correct += torch.sum( torch.argmax(output, dim=1)==target )
        total_items += input.shape[0]

    # calculate average loss/accuracy per batch
    train_loss = total_loss / len(train_loader) # len(train_loader) is number of minibatch
    accuracy = total_correct / total_items # total_items will be total number of data points in train_dataset
    
    # returns tuple of Loss_ave, Accuracy
    return train_loss.item(), accuracy.item()

def test(model, device, test_loader, criterion): # for both validation and test
    '''
    Function for testing our models. One call to test() runs through every
    datapoint in our dataset once.
    '''
    # set model to evaluation mode
    model.eval()

    # we'll keep track of total loss to calculate the average later
    test_loss = 0
    total_correct = 0
    total_items = 0

    #don't perform backprop if testing
    with torch.no_grad():
        # iterate thorugh each test image
        for (input,target) in test_loader:

            # send input to GPU if using GPU
            input = input.double()# both model and data use double precision
            input = input.to(device)
            target = target.type(torch.LongTensor).to(device)

            # run input through our model
            output = model(input)

            # calculate Loss
            loss_value = criterion( output, target )

            # Accumulate for accuracy
            test_loss += loss_value
            total_correct += torch.sum( torch.argmax(output, dim=1)==target )
            total_items += input.shape[0]

    # calculate average loss/accuracy per batch
    test_loss /= len(test_loader)
    accuracy = total_correct / total_items

    return test_loss.item(), accuracy.item()



### do the training and cross-validation

D_train_val = topoDataset(data_train_val,label_train_val) # includes D_train, D_val


K = 10 # Divide into K parts for cross-validation
n_val = len(D_train_val)//K + 1 # number of data in one partition
print('data size for train and cross-validation:' +str(label_train_val.shape[0]))
#print('data size for validation:' +str(n_val))

# store all loss and accuracy of each epoch and of all K-folded validation
# epoch = 0 records loss,acc without training yet
train_loss = np.zeros([epochs+1,K]) # training loss
train_acc = np.zeros([epochs+1,K])  # training accuracy
val_loss = np.zeros([epochs+1,K]) # validation loss
val_acc = np.zeros([epochs+1,K])  # validation accuracy

t_start = datetime.now()
print('Start! Start time is: '+ str(t_start))

for k in range(K):
    # get trainning and validation dataset:
    n_start = n_val*k
    n_end = min(n_val*(k+1)+1,N_train_val)
    D_train, D_val = D_train_val.split_val(n_start,n_end)
    train_loader = DataLoader(D_train, batch_size = batchsize, shuffle = True) # data shuffled after one full epoch
    val_loader = DataLoader(D_val, batch_size = batchsize, shuffle = False)
    model.apply(init_normal) # reset model weights
    # training and validation:
    print('start cross-validate part '+str(k+1))
    # epoch = 0 records loss,acc without training yet. so both use test(..)
    train_loss[0,k], train_acc[0,k] = test(model, device, train_loader, criterion)
    val_loss[0,k], val_acc[0,k] = test(model, device, val_loader, criterion)
    for epoch in range(1,epochs+1):
        train_loss[epoch,k], train_acc[epoch,k] = train(model, device, train_loader, criterion, optimizer)
        val_loss[epoch,k], val_acc[epoch,k] = test(model, device, val_loader, criterion)
        if epoch%4 == 0:
            print('Train Epoch: {:02d} \tTrain Loss: {:.6f} \tTrain Acc: {:.6f} \t\t\t \
                  Val Loss: {:.6f} \tVal Acc: {:.6f}\n'\
                  .format(epoch,train_loss[epoch,k], train_acc[epoch,k], val_loss[epoch,k], val_acc[epoch,k]))   
    #print('finished cross-validate part '+str(k+1))

t_end = datetime.now()
print('Time used for cross-validation:' + str(t_end-t_start) )
print('Cross averaged final training accuracy = ' +str(sum(train_acc[-1,:])/K))
print('Cross averaged final validation accuracy = ' +str(sum(val_acc[-1,:])/K))

### Take average and std across K-folds: ###
result_ave = np.zeros([epochs+1,4]) # averaged for all K-fold, axis 1: train_loss, train_acc, val_loss, val_acc
result_ave[:,0] = np.mean(train_loss,axis=1)
result_ave[:,1] = np.mean(train_acc,axis=1)
result_ave[:,2] = np.mean(val_loss,axis=1)
result_ave[:,3] = np.mean(val_acc,axis=1)
result_std = np.zeros([epochs+1,4]) # standard deviation across K-fold, axis 1: train_loss, train_acc, val_loss, val_acc
result_std[:,0] = np.std(train_loss,axis=1)
result_std[:,1] = np.std(train_acc,axis=1)
result_std[:,2] = np.std(val_loss,axis=1)
result_std[:,3] = np.std(val_acc,axis=1)


### Save the results: ###


folderMother = 'pytorch neural network ReLU CrossEntropyLoss Adam'
folderChild = 'unit{:s}_batch{:.0f}_lr{:.0e}_betas{:s}_eps{:.0e}_decay{:.1f}_amsgrad{:s}_epoch{:.0f}'\
        .format(str(units),batchsize,lrate,str(betas),eps,decay,str(amsgrad),epochs)
headerName = 'unit={:s},batch={:.0f},lr={:.0e},betas={:s},eps={:.0e},decay={:.1f},amsgrad={:s},epoch={:.0f}'\
        .format(str(units),batchsize,lrate,str(betas),eps,decay,str(amsgrad),epochs)
figure_head = 'unit={:s},batch={:.0f},lr={:.0e},betas={:s},eps={:.0e},decay={:.1f},amsgrad={:s}'\
        .format(str(units),batchsize,lrate,str(betas),eps,decay,str(amsgrad))
'''
folderMother = 'pytorch linear ReLU CrossEntropyLoss Adam'
folderChild = 'batch{:.0f}_lr{:.0e}_betas{:s}_eps{:.0e}_decay{:.1f}_amsgrad{:s}_epoch{:.0f}'\
        .format(batchsize,lrate,str(betas),eps,decay,str(amsgrad),epochs)
headerName = 'batch={:.0f},lr={:.0e},betas={:s},eps={:.0e},decay={:.1f},amsgrad={:s},epoch={:.0f}'\
        .format(batchsize,lrate,str(betas),eps,decay,str(amsgrad),epochs)
figure_head = 'batch={:.0f},lr={:.0e},betas={:s},eps={:.0e},decay={:.1f},amsgrad={:s}'\
        .format(batchsize,lrate,str(betas),eps,decay,str(amsgrad))


'''
SaveOrNot = True
if SaveOrNot == True:
    codeDir = os.path.dirname(os.path.realpath('__file__'))
    fileDir = os.path.join(codeDir,'result\\{:s}\\{:s}'.format(folderMother,folderChild))
    head = '{:s}:\n {:s}'.format(folderMother,headerName)
    if not os.path.exists(fileDir):
        os.makedirs(fileDir) 
    with open(os.path.join(fileDir,'train_loss.txt'),'w') as f: # note: will overwrite
            np.savetxt(f, train_loss,fmt='%.8f',header=head)
    with open(os.path.join(fileDir,'val_loss.txt'),'w') as f: # note: will overwrite
            np.savetxt(f, val_loss,fmt='%.8f',header=head)
    with open(os.path.join(fileDir,'train_acc.txt'),'w') as f: # note: will overwrite
            np.savetxt(f, train_acc,fmt='%.8f',header=head)
    with open(os.path.join(fileDir,'val_acc.txt'),'w') as f: # note: will overwrite
            np.savetxt(f, val_acc,fmt='%.8f',header=head)
    with open(os.path.join(fileDir,'result_ave.txt'),'w') as f: # note: will overwrite
            np.savetxt(f, result_ave,fmt='%.8f',header=head+'\ntrain_loss, train_acc, val_loss, val_acc')
    with open(os.path.join(fileDir,'result_std.txt'),'w') as f: # note: will overwrite
            np.savetxt(f, result_std,fmt='%.8f',header=head+'\ntrain_loss, train_acc, val_loss, val_acc')

### plot: ###
PlotOrNot = True
if PlotOrNot == True :
    fig, axs = plt.subplots(2, 2,constrained_layout=True)
    figure_title = '{:s}:\n {:s}'.format(folderMother,figure_head)
    fig.suptitle(figure_title)
    axs[0,0].set(ylabel='Training Loss',ylim=[0.1,0.8])
    axs[0,1].set(ylabel='Validation Loss',ylim=[0.1,0.8])
    axs[1,0].set(xlabel='epoch',ylabel='Training Accuracy',ylim=[0.4,0.98])
    axs[1,1].set(xlabel='epoch',ylabel='Validation Accuracy',ylim=[0.4,0.98])
    for k in range(K):
        axs[0,0].plot(range(epochs+1),train_loss[:,k], label = 'train '+str(k+1))
        axs[0,1].plot(range(epochs+1),val_loss[:,k], label = 'val '+str(k+1))
        axs[1,0].plot(range(epochs+1),train_acc[:,k], label = 'train '+str(k+1))
        axs[1,1].plot(range(epochs+1),val_acc[:,k], label = 'val '+str(k+1))
    axs[0,0].legend(fontsize='xx-small')    
    #axs[0,1].legend()
    #axs[1,0].legend()
    #axs[1,1].legend()
    #plt.tight_layout()
    if SaveOrNot == True:
        fig.savefig(os.path.join(fileDir,'figure'),dpi=1024)
    fig.show()


### Retrain the model on whole D_train_val dataset then test with preserved D_test

TestOrNot = False
if TestOrNot == True:
    
    cutoff_epoch = 20 # set after analyzed cross-validation results
    print('final cutoff epoch = '+ str(cutoff_epoch))

    data_test = np.loadtxt(os.path.join(dataDir,'data_test.txt'), unpack=False) # (N_test, D)
    label_test = np.loadtxt(os.path.join(dataDir,'label_test.txt'), unpack=False) # (N_test, )

    D_test = topoDataset(data_test,label_test)
    
    test_loader = DataLoader(D_test, batch_size = batchsize, shuffle = False)
    train_val_loader = DataLoader(D_train_val, batch_size = batchsize, shuffle = False)

    final_test = np.zeros([cutoff_epoch+1,4]) # training loss, training accuracy,test loss,  test accuracy

    model.apply(init_normal) # reset model weights

    t_start = datetime.now()
    
    # epoch = 0 records loss,acc without training yet. so both use test(..)
    final_test[0,0], final_test[0,1] = test(model, device, train_val_loader, criterion)
    final_test[0,2], final_test[0,3] = test(model, device, test_loader, criterion)
    
    for epoch in range(1,cutoff_epoch+1):
        final_test[epoch,0], final_test[epoch,1] = train(model, device, train_val_loader, criterion, optimizer)
        final_test[epoch,2], final_test[epoch,3] = test(model, device, test_loader, criterion)

    t_end = datetime.now()
    print('Time used for final train test:' + str(t_end-t_start) )
    print('final training accuracy = ' +str(final_test[-1,2]))
    print('final test accuracy = ' +str(final_test[-1,3]))

    SaveOrNot = True
    if SaveOrNot == True:
        codeDir = os.path.dirname(os.path.realpath('__file__'))
        fileDir = os.path.join(codeDir,'result\\{:s}\\{:s}'.format(folderMother,folderChild))
        head = '{:s}:\n {:s}'.format(folderMother,headerName)
        if not os.path.exists(fileDir):
            os.makedirs(fileDir) 
        with open(os.path.join(fileDir,'final_test.txt'),'w') as f: # note: will overwrite
                np.savetxt(f, final_test,fmt='%.8f',header=head+'\n training loss,\t training accuracy,\t test loss,\t test accuracy')

    PlotOrNot = True
    if PlotOrNot == True :
        fig, axs = plt.subplots(2, 2,constrained_layout=True)
        figure_title = '{:s}:\n {:s}'.format(folderMother,figure_head)
        fig.suptitle(figure_title)
        axs[0,0].set(ylabel='Training Loss',ylim=[0.1,0.8])
        axs[0,1].set(ylabel='Test Loss',ylim=[0.1,0.8])
        axs[1,0].set(xlabel='epoch',ylabel='Training Accuracy',ylim=[0.4,0.98])
        axs[1,1].set(xlabel='epoch',ylabel='Test Accuracy',ylim=[0.4,0.98])
        axs[0,0].plot(range(cutoff_epoch+1),final_test[:,0]) #training loss
        axs[0,1].plot(range(cutoff_epoch+1),final_test[:,2]) #test loss
        axs[1,0].plot(range(cutoff_epoch+1),final_test[:,1]) #training acc
        axs[1,1].plot(range(cutoff_epoch+1),final_test[:,3]) #test acc
        #axs[0,0].legend(fontsize='xx-small')    
        #axs[0,1].legend()
        #axs[1,0].legend()
        #axs[1,1].legend()
        #plt.tight_layout()
        if SaveOrNot == True:
            fig.savefig(os.path.join(fileDir,'figureTest'),dpi=1024)
        fig.show()

    ''' linear classifier case:
    trained_weights = model[0].weight.detach().numpy().T
    ind = trained_weights[:,1].argsort() # from lowest to highest
    print(trained_weights[ind[-10:],1])
    with open(os.path.join(fileDir,'linear_weights.txt'),'w') as f:
	np.savetxt(f, trained_weights,fmt='%.8f',header=head+'\n trivial, non-trivial')
    '''
        
