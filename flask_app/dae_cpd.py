#import libraries

import numpy as np

import scipy.io as sio
from scipy.signal import savgol_filter, find_peaks

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

import sklearn.metrics
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 
from itertools import cycle, tee


#parameters for saving model
print_every=10
save_every=10


##################################################################################

								#Data Preparation#

##################################################################################

def window(data, idx, window_size, dim):
    return data.reshape((len(idx), window_size*dim))

def train_test(idx, window_size, Y, L, D): 
    n = len(idx)
    #g = n-1
    #print(n)
    
    A = torch.zeros((n, D))  
    #print(A.size())
    B = torch.zeros((n, 1))
    #print(B.size())
    
    X_p = torch.zeros((n, window_size, D)) ## past samples sequence set created by sliding window
    #print(X_p.size())
    X_f = torch.zeros((n, window_size, D)) ## future samples sequence set created by sliding window
    #print(X_f.size())
    
    for i in range(n):
        l = idx[i] - window_size
        #print(l)
        m = idx[i]
        #print(m)
        n = idx[i] + window_size
        #print(n)


        X_p[i, :, :] = torch.from_numpy(Y[l:m, :])
        X_f[i, :, :] = torch.from_numpy(Y[m:n, :])
            
        A[i, :] = torch.from_numpy(Y[m, :])
        
        B[i] = torch.from_numpy(L[m])
            

    #data, future_data, true_data, labels = Variable(X_p), Variable(X_f), Variable(A), Variable(B)
    X_p =  window(X_p, idx, window_size, D) 
    X_f =  window(X_f, idx, window_size, D)
    
    data, future_data, true_data, labels = Variable(X_p), Variable(X_f), Variable(A), Variable(B)
    
    return data, future_data, true_data, labels


def prepare_data(Y, L, window_size = 25, trn_ratio = 0.60, val_ratio=0.80):
    
    T, D = Y.shape                  # T: time length; D: number of variables
    
    n_trn = int(np.ceil(T * trn_ratio)) # splitting point between train set and validation set 
    n_val = int(np.ceil(T * val_ratio))  # splitting point between validation set and test set

    n_tst = T-window_size
    print('Length of dataset:', T, 'Number of variables:', D, 'First index of validation set:', n_trn, 'First index of test set:', n_val)

    #print(n_tst)

    train_set_idx = range(window_size, n_trn)
    val_set_idx = range(n_trn, n_val)
    test_set_idx = range(n_val, n_tst)

    '''
    print('number_of_training_samples:', len(train_set_idx), 'number_of_validation_samples:', len(val_set_idx), 'number_of_test_samples:', len(test_set_idx)) ## number of train samples, number of validation samples, number of test samples
    '''
    return train_set_idx, val_set_idx, test_set_idx


##########################################################################################################

                                        ###Network###

##########################################################################################################


class encoder (nn.Module):
    def __init__(self, x_dim, z_dim):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 40),  
            nn.ReLU(),
            nn.Linear(40, 30),     
            nn.ReLU(),
            nn.Linear(30, 20),     
            nn.ReLU(),
            nn.Linear(20, z_dim))
            
    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x
    
    
class decoder (nn.Module):
    def __init__(self, x_dim, z_dim):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 40), 
            nn.ReLU(),
            nn.Linear(40, x_dim))
        
    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x
    
    
class autoencoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(autoencoder, self). __init__()
        self.Encoder = encoder(x_dim, z_dim)
        self.Decoder = decoder(x_dim, z_dim)
        
    def forward(self, x):
        z = self.Encoder(x)
        x_hat = self.Decoder(z)
        
        return z, x_hat


#############################################################################################################

                                    #Helper functions#

#############################################################################################################


# for stitching train_val_test into complete data for fit_predict
def stitch_data(p, q, r):
    directory = list(p) + list(q) + list(r)
    #print(len(directory))
    
    return directory


# for normalizing data for visualization purpose

def normalize(data):
    scaler = StandardScaler()
    transformed = scaler.fit(data)
    transformed = scaler.transform(data)
    transformed = transformed.flatten()
    return transformed


# smoothing filter, finding peak indexes

def change_finder(x):
    
    x = np.expand_dims(x, axis=1)
    x = normalize(x)
    x = savgol_filter(x, 99 , 3)
    peaks, _ = find_peaks(x, distance=50, width=10)
    
    return x, peaks

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# module for kernel compiuting

def compute_kernel(x, y):    ## add other kernels : cauchy, laplace
    x_size = x.size(0)
    y_size = y.size(0)
    #print('X shape', x.size())
    #print('Y shape', y.size())
    
    dim = x.size(1) 
    
    kernel_input = (x - y).pow(2).mean(1)/float(dim) ## Gaussian Kernel
    
    #print('KERNEL:', kernel_input.size())
    
    return torch.exp(-kernel_input)
    

## computes mmd loss between current and future sample

def _mmd_loss(X_p, X_f):        
    #X_p, X_f = self._past_future(x)
    p_kernel = compute_kernel(X_p, X_p)
    #print('P-kernel:', p_kernel)
    f_kernel = compute_kernel(X_f,X_f)
    #print('F-kernel:', f_kernel.size())
    pf_kernel = compute_kernel(X_p,X_f)
    #print('PF-kernel:', pf_kernel.size())
    mmd = p_kernel + f_kernel - 2*pf_kernel     
    #print('MMD:', mmd.size())				## size of batch size
    
    return mmd
    
    
## Validation using prediction scores with validation ground truths

def valid_epoch(labels_pred, labels):   ## add other metrics
    labels_pred = labels_pred.detach().numpy()  ## per sample mmd scores from training
    #print(labels_pred.shape)
    
    labels_true = labels             ## training labels
    #print(len(labels))
    
    fp_list, tp_list, thresholds = sklearn.metrics.roc_curve(labels_true, labels_pred)
    auc = sklearn.metrics.auc(fp_list, tp_list)
    
    return auc


##load saved model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model




##########################################################################################

                                ###Training + Validation###

##########################################################################################


#train
def train(epoch, train_loader, train_loader_future, labels_train, optimizer, model, criterion, beta):
    
    total_loss, total_loss_p, total_loss_f, total_loss_k =0, 0, 0, 0
    total_mmd = []
    
    for batch_idx, (data, future_data) in enumerate(zip(train_loader, train_loader_future)):
        
        #print('data', data.size())
        
        optimizer.zero_grad()
        z_p, x_hat_p =  model(data)
        z_f, x_hat_f = model(future_data)
        
        #print('encoded', z.size())
        #print('decoded', x_hat.size())
        
        loss_p = criterion(data, x_hat_p)
        loss_f = criterion(future_data, x_hat_f)
    
        mmd = _mmd_loss(z_p, z_f)

        #print('MMD:', mmd)
        loss_m = mmd.mean()
        loss = loss_p + loss_f + beta*loss_m
        
        loss.backward()
        optimizer.step()
        
        total_loss_p +=loss_p.data
        total_loss_f +=loss_f.data
        total_loss +=loss.data
        
        total_mmd.append(mmd)

    
    mmd_score = torch.cat(total_mmd, dim=0)
    #print('all_mmd', mmd_score.size())

    labels_pred = mmd_score
    
    #auc = valid_epoch(labels_pred, labels_train)
    mmd_mean = mmd_score.mean().detach().numpy()
    #print('Train mmd:', mmd_mean)
    
    '''
    if epoch % print_every == 1:
        print("Train Epoch: {} Reconstruction Loss (P) is {:.4f} \t Reconstruction Loss (F) is {:.5f} \t Total Loss: {:0.5f} \t MMD: {:0.5f} \t AUC: {:.5f}".format(epoch +1, total_loss_p/len(train_loader.dataset), total_loss_f/len(train_loader.dataset), total_loss/len(train_loader.dataset), mmd_mean, auc))
    '''
    
    return labels_pred


#validate
def valid(epoch, valid_loader, valid_loader_future, labels_valid, optimizer, model, criterion, beta):
    
    total_loss, total_loss_p, total_loss_f, total_loss_k =0, 0, 0, 0
    total_mmd = []
    
    for batch_idx, (data, future_valid) in enumerate(zip(valid_loader, valid_loader_future)):
        
        z_p, x_hat_p =  model(data)
        z_f, x_hat_f = model(future_valid)
        
        loss_p = criterion(data, x_hat_p)
        loss_f = criterion(future_valid, x_hat_f)
        mmd = _mmd_loss(z_p, z_f)
        loss_m = mmd.mean()
        
        loss = loss_p + loss_f + beta*loss_m
        
        total_loss_p +=loss_p.data
        total_loss_f +=loss_f.data
        total_loss += loss.data
      
        total_mmd.append(mmd)
    
    mmd_score = torch.cat(total_mmd, dim=0)
    #print(mmd_score.size())
    
    labels_pred = mmd_score
    
    #auc = valid_epoch(labels_pred, labels_valid)
    mmd_mean = mmd_score.mean().detach().numpy()
    
    '''
    if epoch % print_every == 1:
        print("Valid Epoch: {} Reconstruction Loss (P) is {:.4f} \t Reconstruction Loss (F) is {:.5f} \t  Total Loss: {:0.5f} \t MMD: {:0.5f} \t AUC: {:.5f}".format(epoch +1, total_loss_p/len(valid_loader.dataset), total_loss_f/len(valid_loader.dataset), total_loss/len(valid_loader.dataset), mmd_mean, auc))
    '''
   
    return labels_pred, mmd_mean
 


# test 
def test(model, test_loader, test_loader_future, labels_test):
    
    total_loss, total_loss_p, total_loss_f, total_loss_k  =0, 0, 0, 0
    total_mmd = []
    
    for batch_idx, (data, future_test) in enumerate(zip(test_loader, test_loader_future)):
        
        z_p, x_hat_p =  model(data)
        z_f, x_hat_f = model(future_test)
        
        mmd = _mmd_loss(z_p, z_f)
          
        #print(mmd.size())
        total_mmd.append(mmd)
    
    mmd_score = torch.cat(total_mmd, dim=0)
    
   
    labels_pred = mmd_score
    
    #auc = valid_epoch(labels_pred, labels_test)
   
    #print('Test mmd:', mmd_mean)
    
    return labels_pred


###################################################################################

									#DAE Change point detection class#

###################################################################################

class dae(object):
    def __init__(self, Y, L, window_size = 25, trn_ratio = 0.60, val_ratio = 0.80, z_dim = 3, epochs = 1000, batch_size = 128, learning_rate = 1e-5, seed =100, beta = 1):

        
        self.Y = Y                              # Signal => shape (n_samples, n_features)
        self.L = np.expand_dims(L, axis = 1)    # Labels => shape (n_samples, 1)
        self.T, self.D = Y.shape
        self.window_size = window_size        # length of sliding window
        self.x_dim = self.window_size*self.D    # input dimension for autoencoder
        self.z_dim = z_dim						# encoding dimension for autoencoder
        self.trn_ratio = trn_ratio				# % of data used for training
        self.val_ratio = val_ratio				# (val_ratio-trn_ratio) => % of data used for validation, remaining is test set
        self.epochs = epochs					
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.print_every =10
        self.save_every=10
        self.seed = seed
        self.beta = beta                        # default is 1
        
        # find indexes corresponding to train_val_test set
        self.trn_idx, self.val_idx, self.tst_idx = prepare_data(self.Y, self.L, self.window_size, self.trn_ratio, self.val_ratio)
        
        # create train_val_test set by sliding windows
        self.train_dataset, self.future_train, self.true_train, self.labels_train = train_test(self.trn_idx, self.window_size, self.Y, self.L, self.D)
        self.valid_dataset, self.future_valid, self.true_valid, self.labels_valid = train_test(self.val_idx, self.window_size, self.Y, self.L, self.D)
        self.test_dataset, self.future_test, self.true_test, self.labels_test = train_test(self.tst_idx, self.window_size, self.Y, self.L, self.D)

        '''
        print('Shape of Sequenced Samples for Training:', self.train_dataset.size())
        print('Shape of Sequenced Samples for Validation:', self.valid_dataset.size())
        print('Shape of Sequenced Samples for Testing:', self.test_dataset.size())
        print('Shape of Sequenced Future Samples for Training:', self.future_train.size())
        print('Shape of Sequenced Future Samples for Validation:', self.future_valid.size())
        print('Shape of Sequenced Future Samples for Testing:', self.future_test.size())
        print('Shape of Original Train Data:', self.true_train.size())
        print('Shape of Original Validation Data:', self.true_valid.size())
        print('Shape of Original Test Data:', self.true_test.size())
        print('Shape of Training Labels:', self.labels_train.size())
        print('Shape of Validation Labels:', self.labels_valid.size())
        print('Shape of Testing Labels:', self.labels_test.size())
        '''

        ##dataloaders
        ##Current

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last = False)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, drop_last = False)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last = False)


        ##Future
        self.train_loader_future = torch.utils.data.DataLoader(self.future_train, batch_size=self.batch_size, shuffle=False, drop_last = False)
        self.valid_loader_future = torch.utils.data.DataLoader(self.future_valid, batch_size=self.batch_size, shuffle=False, drop_last = False)
        self.test_loader_future = torch.utils.data.DataLoader(self.future_test, batch_size=self.batch_size, shuffle=False, drop_last = False)


        ## Labels
        self.labels_loader_train = torch.utils.data.DataLoader(self.labels_train, batch_size=self.batch_size, shuffle=False, drop_last = False)
        self.labels_loader_valid = torch.utils.data.DataLoader(self.labels_valid, batch_size=self.batch_size, shuffle=False, drop_last = False)
        self.labels_loader_test= torch.utils.data.DataLoader(self.labels_test, batch_size=self.batch_size, shuffle=False, drop_last = False)
        
        
        ## model
        torch.manual_seed(self.seed)
  
        self.model = autoencoder(self.x_dim, self.z_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)


        
    # fit method
    def fit(self):
        #best_val_auc = -1
        best_mmd = -1
        best_epoch = -1
        
        
        for epoch in range(self.epochs):
    
            #training + validation
            train_score = train(epoch, self.train_loader, self.train_loader_future, self.labels_train, self.optimizer, self.model, self.criterion, self.beta)
            val_score, mmd = valid(epoch, self.valid_loader, self.valid_loader_future, self.labels_valid, self.optimizer, self.model, self.criterion, self.beta)

            #if val_auc > best_val_auc:
            if mmd > best_mmd:
                #best_val_auc = val_auc
                best_epoch = epoch
                best_mmd = mmd
            
                #saving model
                self.fn = 'ae_state_dict_'+ 'best_epoch'+'.pth'
                self.checkpoint = { 'model': autoencoder(self.x_dim, self.z_dim), 'state_dict': self.model.state_dict(), 'optimizer' : self.optimizer.state_dict()}
                self.best = torch.save(self.checkpoint, self.fn)
        '''
        print("Best Epoch:", best_epoch)
        print("Best AUC:", best_val_auc)
        '''
        print("Model Creation Complete!")

        return self
    
    
    # predict method, for result on unseen tets data
    def predict(self):
        model = load_checkpoint(self.fn)
        self.mmd_score = test(model, self.test_loader, self.test_loader_future, self.labels_test)
        
        self.score, self.pred_pts = change_finder(self.mmd_score)

        return self.pred_pts
    

    def complete_set(self):
        self.present = stitch_data(self.train_dataset, self.valid_dataset, self.test_dataset)
        self.future = stitch_data(self.future_train, self.future_valid, self.future_test)
        self.label = stitch_data(self.labels_train, self.labels_valid, self.labels_test)
        
        #print(len(self.present))
        #print(len(self.future))
        #print(len(self.label))
        
        return self.present, self.future, self.label
    

    # fit_predict method, for result on full dataset
    def fit_predict(self):
        model = load_checkpoint(self.fn)
        
        self.present, self.future, self.label = self.complete_set()
        
        #print(len(self.label))
        
        self.data = torch.utils.data.DataLoader(self.present, batch_size=self.batch_size, shuffle=False, drop_last = False)
        self.future_data = torch.utils.data.DataLoader(self.future, batch_size=self.batch_size, shuffle=False, drop_last = False)
        self.label_set = torch.utils.data.DataLoader(self.label, batch_size=self.batch_size, shuffle=False, drop_last = False)
        
        #print(len(self.label_set))
        self.mmd_score = test(model, self.data, self.future_data, self.label)
        
        self.score, self.pred_pts = change_finder(self.mmd_score)
        
        
        return self.pred_pts
        
        
    # display signal, predicted and true chnage points, function taken from 
    #ruptures repo: https://github.com/deepcharles/ruptures/blob/master/ruptures/show/display.py

    
    def display(self, signal, true_chg_pts, computed_chg_pts=None, **kwargs):
        
        COLOR_CYCLE = ["#4286f4", "#f44174"]
        
        #true_chg_pts = true_chg_pts[self.window_size: -self.window_size]

        if type(signal) != np.ndarray:
            # Try to get array from Pandas dataframe
            signal = signal.values

        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        n_samples, n_features = signal.shape

        # let's set a sensible defaut size for the subplots
        matplotlib_options = {"figsize": (10, 2 * n_features), } # figure size}
        # add/update the options given by the user
        matplotlib_options.update(kwargs)

        # create plots
        fig, axarr = plt.subplots(n_features, sharex=True, **matplotlib_options)
        if n_features == 1:
            axarr = [axarr]
        
        for axe, sig in zip(axarr, signal.T):
            color_cycle = cycle(COLOR_CYCLE)
            # plot s
            axe.plot(range(n_samples), sig)

            # color each (true) regime
            bkps = [0] + sorted(true_chg_pts)
            #print(bkps)
            alpha = 0.2  # transparency of the colored background
            
            for (start, end), col in zip(pairwise(bkps), color_cycle):
                axe.axvspan(max(0, start - 0.5),
                        end - 0.5,
                        facecolor=col, alpha=alpha)
            
            color = "k"  # color of the lines indicating the computed_chg_pts
            linewidth = 3  # linewidth of the lines indicating the computed_chg_pts
            linestyle = "--"  # linestyle of the lines indicating the computed_chg_pts
            # vertical lines to mark the computed_chg_pts
            if computed_chg_pts is not None:
                for bkp in computed_chg_pts:
                    if bkp != 0 and bkp < n_samples:
                        axe.axvline(x=bkp - 0.5,
                                    color=color,
                                    linewidth=linewidth,
                                    linestyle=linestyle)
            

        fig.tight_layout()

        return fig, axarr

        




