import os
import math
import torch
import pickle
import ForecastLib
import numpy as np
import pandas as pd
import padasip as pa
from itertools import product
from sklearn import preprocessing
from hyperopt import fmin, tpe, hp
from RecRVFL_.RecRVFL import RecRVFL
###############################################################################

def format_data(data, order, idx=0):
    n_sample = data.shape[0]-order
    x = np.zeros((n_sample, data.shape[1]*order))
    y = np.zeros((n_sample, 1))
    for i in range(n_sample):
        x[i,:] = data[i:i+order,:].ravel()
        y[i] = data[i+order,idx]
    return x, y

###############################################################################

def select_indexes(data, indexes):
    return data[indexes,:]

###############################################################################
    
def compute_error(actuals, predictions, history=None):
    actuals = actuals.ravel()
    predictions = predictions.ravel()
    
    metric = ForecastLib.TsMetric()
    
    error={}
    error['MAE'] = metric.MAE(actuals, predictions)
    error['MAPE']  =metric.MAPE(actuals, predictions)
    error['RMSE'] = metric.RMSE(actuals, predictions)
    
    
    if history is not None:
        history = history.ravel()
        error['MASE'] = metric.MASE(actuals, predictions, history)
    return error

###############################################################################

def get_data(name):
    file_name = name+'.csv'
    dat = pd.read_csv(file_name)
    dat = dat.fillna(method='ffill')
    return dat,dat.columns

###############################################################################

class Struct(object): pass

###############################################################################
def config_load(iss,IP_indexes):

    configs = Struct()
    configs.iss = iss # set insput scale 0.1 for all recurrent layers
    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 0 # activate pre-train
    configs.enhConf = Struct()
    configs.enhConf.connectivity = 1 # connectivity of recurrent matrix
    configs.readout = Struct()
    configs.readout.trainMethod = 'Ridge' # train with singular value decomposition (more accurate)
    
    return configs 

###############################################################################

def dRecRVFL_predict(hyper, data, train_idx, test_idx, layer, s, last_states=None, dec_=None, alldecs=None):

    np.random.seed(s) # random seed
    Nu=data.inputs.shape[0] # number of input features

    Nh = hyper[0][0] # number of hidden neurons
    Nl = layer # Layer
    
    #defining the list for regularization and input scale parameters
    reg, iss = [], []
    
    for h in hyper:
        reg.append(h[1])        
        iss.append(h[2])
    
    layers=hyper[0][3]
 
    trainX = select_indexes(data.inputs, train_idx)
    train_targets = select_indexes(data.targets, train_idx)
    
    if Nl == 1:
        last_states = data.inputs
        
    redrvfl = RecRVFL(Nl-1, last_states.shape[1], trainX.shape[1], Nh, lamb = reg[-1], input_scale = iss[-1], layers = layers, device = 'cpu')
    
    redrvfl.train()
    redrvfl.eval()
    
    # print(torch.Tensor(data.inputs).float())
    
    if Nl == 1:
        with torch.no_grad():
            states = redrvfl.transform(X_raw = torch.Tensor(data.inputs).float()).detach().clone()
    else:
        with torch.no_grad():
            states = redrvfl.transform(X_raw = torch.Tensor(data.inputs).float(), X = torch.Tensor(last_states).float()).detach().clone()
            
    #beta weight calculation
    redrvfl.init_weight(torch.Tensor(select_indexes(last_states, train_idx)).float(), torch.Tensor(train_targets).float(), torch.Tensor(trainX).float())    
    
    if Nl == 1:
        with torch.no_grad():
            preds = redrvfl(X_raw = torch.Tensor(data.inputs).float()).detach().clone()
    else:
        with torch.no_grad():
            preds = redrvfl(X_raw = torch.Tensor(data.inputs).float(), X = torch.Tensor(last_states).float()).detach().clone()
            
    outputs = select_indexes(preds, test_idx)
    return outputs, states[:,:]

###############################################################################

def redrvfl_predict(hyper,data,train_idx,test_idx,s):
    
    np.random.seed(s) # random seed

    Nr = hyper[0][0] # number of hidden neurons
    Nl = len(hyper) # number of recurrent layers
    
    #defining the list for regularization and input scale parameters
    reg, iss = [], []
    
    for h in hyper:
        reg.append(h[1])        
        iss.append(h[2])
    layers=hyper[0][3]

        
    last_states = None
    outputs = np.zeros((len(test_idx),Nl))
    
    trainX = select_indexes(data.inputs, train_idx)
    train_targets = select_indexes(data.targets, train_idx)
    
    for l in range(Nl):

        if l==0:
            last_states = data.inputs

        redrvfl = RecRVFL(l, last_states.shape[1], trainX.shape[1], Nr, lamb = reg[-1], input_scale = iss[-1], layers = layers, device = 'cpu')
        redrvfl.train()
        redrvfl.eval()
        if l == 0:
            with torch.no_grad():
                states = redrvfl.transform(X_raw = torch.Tensor(data.inputs).float()).detach().clone()
        else:
            with torch.no_grad():
                states = redrvfl.transform(X_raw = torch.Tensor(data.inputs).float(), X = torch.Tensor(last_states).float()).detach().clone()
               
        #beta weight calculation
        redrvfl.init_weight(torch.Tensor(select_indexes(last_states, train_idx)).float(), torch.Tensor(train_targets).float(), torch.Tensor(trainX).float())
        
        if l == 0:
            with torch.no_grad():
                preds = redrvfl(X_raw = torch.Tensor(data.inputs).float())
        else:
            with torch.no_grad():
                preds = redrvfl(X_raw = torch.Tensor(data.inputs).float(), X = torch.Tensor(last_states).float()).cpu().detach().numpy()
        
        test_outputs_norm = select_indexes(preds, test_idx)
        outputs[:,l:l+1]=test_outputs_norm
        last_states = states


    return np.median(outputs,axis=1).reshape(-1,1), outputs#outputs.mean(axis=1).reshape(-1,1)
###################################################################################################################

def cross_validation(hypers,data,raw_data,train_idx,val_idx,Nl,regs,input_scale,layers,scaler=None,s=0,boat=50):
    
    #defining an empty list to store the best hyperparameters corresponding to each layer
    best_hypers = [] 
    np.random.seed(s)  # random seed
    layer_s = None # intialising the state layer to None
    for i in range(Nl):

        layer = i+1
        layer_h,layer_s = layer_cross_validation(hypers,data,raw_data,train_idx,val_idx,layer,
                           scaler=scaler,s=s,last_states=layer_s,best_hypers=best_hypers.copy(),boat=boat)

        Nhs=[layer_h[0]]  #number of hidden units corresponding to each optimized layer
        if layer==1:
            hypers=list(product(Nhs,regs,input_scale,layers))   
            
        best_hypers.append(layer_h)

    return best_hypers

###############################################################################################################################

def layer_cross_validation(hypers,data,raw_data,train_idx,val_idx,layer,
                           scaler=None,s=0,last_states=None,best_hypers=None,boat=50):

    np.random.seed(s)   #random seed

    space={'layer' : hp.choice('layer', [layer]),
           'data' : hp.choice('data', [data]),
           'raw_data' : hp.choice('raw_data', [raw_data]),
           'last_states' : hp.choice('last_states', [last_states]),
           'scaler' : hp.choice('scaler', [scaler]),
           's' : hp.choice('s', [s]),
           'val_idx' : hp.choice('val_idx', [val_idx]),
           'train_idx' : hp.choice('train_idx', [train_idx]),
           'best_hypers' : hp.choice('best_hypers', [best_hypers]),
            'input_scale' : hp.uniform('input_scale', 0, 1),
            'layers': hp.randint('layers', 1, 4),#'layers': hp.randint('layers', 1, 4),
            'regs' : hp.uniform('regs', 0, 1)}
    
    
    if layer==1:
        space['Nhs']=hp.randint('Nhs', 20, 100)#space['Nhs']=hp.randint('Nhs', 20, 200)
    else:
        best_hidden=[best_hypers[0][0]]
        space['Nhs']=hp.randint('Nhs', 20, 100)#space['Nhs']=hp.randint('Nhs', 20, 200)
        
    # defining the hyperopt optimization function
    args=fmin(fn=layer_obj,
                space=space,
                max_evals=boat,
                rstate=np.random.default_rng(seed=0),
                algo=tpe.suggest)
    
    if layer==1:
        best_hyper=[args['Nhs'],args['regs'],args['input_scale'],args['layers']]
    else:
        best_hyper=[best_hidden[0],args['regs'],args['input_scale'],args['layers']]
        
    if layer>1:
            hyper_=best_hypers.copy()
            hyper_.append(best_hyper)
    else:
        hyper_=[best_hyper]
        
    _,best_state=dRecRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                         s,last_states=last_states)
   
    return best_hyper,best_state

###############################################################################################################################

def layer_obj(args):
    layer=args['layer']
    best_hypers=args['best_hypers']

    hyper=[args['Nhs'],args['regs'],args['input_scale'],args['layers']]
    data=args['data']
    train_idx,val_idx=args['train_idx'],args['val_idx']
    scaler=args['scaler']
    s=args['s']
    raw_data,last_states=args['raw_data'],args['last_states']
    
    if layer>1:
            hyper_=[i for i in best_hypers]
            hyper_.append(hyper)
    else:
        hyper_=[hyper]


    test_outputs_norm,_=dRecRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                     s,last_states=last_states)
    test_outputs=scaler.inverse_transform(test_outputs_norm)
    actuals=raw_data[-len(val_idx):]
    test_err=compute_error(actuals,test_outputs,None)
    
    return test_err['RMSE']

####################################################################################################################################
Nhs=np.arange(50,300,50) #number of hidden neurons

Nls=[10] #number of hidden layers
regs=[0] #value of the regularization parameters
input_scale=[0.1] #value of the input scale parameters
layers = [1] #number of layers
deepRecRVFL_hypers=list(product(Nhs,regs,input_scale,layers))
order=1 #look back/n_past
# seeds=6 #choose the seed
stocks = ["DJI"]
boat=100 #number of epochs/trials in hyperopt
for st in stocks:
    
        results = []
        test_pres_ea=[]

        #loading the dataframe corresponding to the month,year,country, and year

        df_data = pd.read_csv(f"datasets//{st}.csv")
        data_=df_data['Close'].values.reshape(-1,1)
        
        #defining the min-max scaler
        scaler=preprocessing.MinMaxScaler() 
        allpres = []
        for s in np.arange(seeds-1, seeds):
            np.random.seed(s)  #random seed
            
            #defining the validation length to be 10% an-d test length to be 10%              
            val_l,test_l=int(0.1*data_.shape[0]),int(0.2*data_.shape[0]) 
            
            #fitting the scaler to the training data
            scaler.fit(data_[:-test_l-val_l])
            #transforming the full data into scaled full data w.r.t. train_data based scaler
            norm_data=scaler.transform(data_)
            
            #Structering the data
            data=Struct()
            data.inputs,data.targets=format_data(norm_data,order)
            
            #defining the training data length
            train_l=data.inputs.shape[0]-val_l-test_l
            
            #defining the indexes corresponding to the train,validation, test datasets
            train_idx=range(train_l)
            val_idx=range(train_l,train_l+val_l)
            test_idx=range(train_l+val_l,data.inputs.shape[0])
                    

            ed_best_hypers = cross_validation(deepRecRVFL_hypers[:],data,data_[:-test_l],
                                              train_idx,val_idx,Nls[0],regs,
                                              input_scale,layers,scaler=scaler,s=s,boat=boat)
            #Testing begins
            print('Test')
            
            #defining the full training indexes corresponding to train and validation dataset
            train_idx=range(train_l+val_l)
            
            #fitting the scaler to the test dataset
            scaler.fit(data_[:-test_l])
            #transforming the full data into scaled full data w.r.t. test_data based scaler
            norm_data=scaler.transform(data_)
            
            #Calculating the inputs and targets corresponding to the normalized(scaled) data
            data.inputs,data.targets=format_data(norm_data,order)

            test_outputs_norm_mea,alllayers=redrvfl_predict(ed_best_hypers,data,train_idx,test_idx,s)
            alllayers=scaler.inverse_transform(alllayers)
            allpres.append(alllayers)
            test_outputs_ea=scaler.inverse_transform(test_outputs_norm_mea)
            
            #Appending the test outputs in the list corresponding to each seed
            test_pres_ea.append(test_outputs_ea)
            
            
            actuals=data_[-test_l:]
            history=data_[:-test_l]

            test_err=compute_error(actuals,test_outputs_ea,history)
            print(test_err)
            print(len(ed_best_hypers))
            
            results.append([st, test_err['RMSE'], test_err['MAE'], test_err['MAPE']])
            
        all_p=np.concatenate(allpres,axis=1)
        dfall=pd.DataFrame(all_p)
        dfall.to_csv('ABA/allpRedRVFLBOA'+str(Nls[0])+str(boat)+st+'.csv')

        
        output_loc = f"redrvfl_results//{st}//"

        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
            
        results_df = pd.DataFrame(results, columns=['stock', 'test_rmse', 'test_mae', 'test_mape'])
        results_df.to_csv(f'redrvfl_results//{st}//results.csv')

        test_p=np.concatenate(test_pres_ea,axis=1)
        dfea=pd.DataFrame(test_p)
        dfea.to_csv(f'redrvfl_results//{st}//redrvflBOA{boat}')
            
           