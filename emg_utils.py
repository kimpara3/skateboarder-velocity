import json
import datetime
from multiprocessing import Pool

import numpy as np
import cv2
import pywt
import scipy.io as sio
import torch
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import make_scorer

def load_data(file, sub, index):
    data_dict = sio.loadmat(file)
    arr = data_dict[sub][0][0][index]

    if len(arr.flatten())==1:
        arr=arr.flatten()[0]
    else:
        # (trial,timestep,channel)に変換
        arr=np.transpose(arr,(0,2,1))

    return arr

def prepare_data(file, sub, signal_index=0, stance_index=2):
    signals = load_data(file, sub, signal_index) # (N,T,C)
    stance = load_data(file, sub, stance_index)
    subject = int(sub)-1

    info=np.zeros((signals.shape[0],7)) # [[sub1,sub2,sub3,sub4,sub5,stance,trial]]
    info[:,subject]=1
    info[:,5]=1 if stance=='goofy' else 0
    info[:,6]=np.arange(len(signals))
    
    additional_channel=np.broadcast_to(info[:,None,:], (info.shape[0], signals.shape[1], info.shape[1])) # (N,T,C)

    ret = np.concatenate([signals,additional_channel],axis=2)
    return ret

def prepare_all():
    subs="0001 0002 0003 0004 0005".split()
    train_data={}
    train_target={}

    for sub in subs:
        if sub=="0005":
            train_data[sub] = prepare_data("reference.mat", sub, stance_index=4)
            train_target[sub] = load_data("reference.mat", sub, 1)
        elif sub=="0005r":
            train_data[sub] = prepare_data("reference.mat", "0005", signal_index=2,stance_index=4)
            train_target[sub] = load_data("reference.mat", "0005", 3)
            train_target[sub] = transform_vel(train_target[sub])
        else:
            train_data[sub] = prepare_data("train.mat", sub)
            train_target[sub] = load_data("train.mat", sub, 1)

    return train_data,train_target

def prepare_testdata():
    subs="0001 0002 0003 0004 0005".split()
    test_data={}

    for sub in subs:
        if sub=="0005":
            test_data[sub] = prepare_data("reference.mat", sub, signal_index=2,stance_index=4)
        else:
            test_data[sub] = prepare_data("test.mat", sub)
    return test_data

def transform_vel(vel:np.ndarray)->np.ndarray:
    '''
    vel : x,y,z velocity(N, 30, 3)
    '''
    ret = vel * np.array([-1, -1, 1])
    return ret

def rmse3D(v_true:torch.Tensor, v_pred:torch.Tensor, weight=[1.,1.,1.]):
    '''
    objective function of competition
    v_true,v_pred : tensor(N, 30, 3)
    return : batch_size * rmse
    '''
    weight_tensor = torch.tensor(weight).to(v_true.device)
    dist2 = torch.sum(weight_tensor * (v_true - v_pred) ** 2, axis=2)  # (N, 30)
    mean_dist2 = torch.mean(dist2, axis=1)  # (N,)
    dist = torch.sqrt(mean_dist2)  # (N,)
    batch_total = torch.sum(dist)
    return batch_total

def rmse3D_per_data(y_true:np.ndarray, y_pred:np.ndarray):
    '''
    function for scorer
    '''
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    batch_total = rmse3D(y_true, y_pred).item()
    return batch_total/y_true.shape[0]

rmse3D_for_sk = make_scorer(rmse3D_per_data, greater_is_better=False)

def save_submit(data:dict, filename=None):
    '''
    dictをsubmission形式で保存する
    '''
    submit_dict = {}
    for sub in "0001 0002 0003 0004".split():
        submit_dict.update(make_submit_dict(data[sub],subject=sub))

    if not filename:
        now=datetime.datetime.now()+datetime.timedelta(hours=9)
        filename=now.strftime("submit%m%d%H%M.json")
    with open(filename, "w") as f:
        json.dump(submit_dict, f)
    print(filename)

def make_submit_dict(data, subject)->dict:
    '''
    arrayを一人分のdictにする
    '''
    key=f"sub{int(subject)}"
    submit_dict={key:{}}
    for n in range(len(data)):
        submit_dict[key][f"trial{n+1}"]=data[n].tolist()

    return submit_dict

def make_submit(data, subject)->str:
    '''
    arrayを一人分のjson stringにする
    '''
    json_dict = make_submit_dict(data, subject)
    return json.dumps(json_dict)

def restore_submit(json_str_or_dict:str|dict, subject=None)->np.ndarray:
    '''
    json stringからarrayにする
    subjectが与えられればそのsubject,
    与えられなければ最初のもの
    '''

    if type(json_str_or_dict) == str:
        submit_dict=json.loads(json_str_or_dict)
    else:
        submit_dict=json_str_or_dict

    if subject is None:
        key = list(submit_dict.keys())[0]
    else:
        key = f"sub{int(subject)}"

    mat = list(submit_dict[key].values())
    mat = np.array(mat)
    return mat

def rotate_data(data:np.ndarray, radian:float)->np.ndarray:
    '''
    X,Yを回転させる。Zはそのまま。
    data : shape (N,30,3)
    radian : rotate angle(radian)
    '''
    rotate=np.empty_like(data)
    sn=np.sin(radian)
    cs=np.cos(radian)
    R=np.array([
        [cs,-sn],
        [sn,cs]
    ])
    
    for n in range(len(data)):
        for t in range(30):
            rotate[n,t,:2] = R@data[n,t,:2]
    rotate[:,:,2]=data[:,:,2]
    return rotate

class CwtGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, signal_channels, fs, num_scale, fmin=5, fmax=500, wavelet="cmor1-1", use_sub=False, use_stance=False, use_count=False, n_jobs=8):
        self.signal_channels = signal_channels
        self.fs = fs

        self.wavelet = wavelet
        frequencies=np.logspace(np.log10(fmin/self.fs),np.log10(fmax/self.fs),num_scale)
        self.scales = pywt.frequency2scale(self.wavelet,frequencies)

        self.sampling_period = 1/self.fs
        self.use_sub = use_sub
        self.use_stance = use_stance
        self.use_count = use_count

        self.n_jobs = n_jobs

        self.frequencies=[]

    def fit(self, X, y=None):
        return self

    def _compute_cwt(self,channel):
        coefs,frequncies = pywt.cwt(self._X[:,:,channel], self.scales, self.wavelet, self.sampling_period)
        return coefs,frequncies

    def transform(self, X):
        '''
        X : ndarray(N, t, C+7)
        return : ndarray(N, C, H, W)
        '''
        self._X = X
        with Pool(processes=self.n_jobs) as pool:
            cwt_results = pool.map(self._compute_cwt, range(self.signal_channels))
        
        self.frequencies=cwt_results[0][1]
        cwt_coefs = np.array([c[0] for c in cwt_results])   # (C, H, N, W)
        cwt_coefs = np.abs(cwt_coefs)
        cwt_coefs = np.transpose(cwt_coefs, (2, 0, 1, 3))  # (N, C, H, W)

        if self.use_sub:
            start_col = self.signal_channels
            end_col = start_col+5

            info = X[:,0,start_col:end_col]
            info_expanded = info[:, :, np.newaxis, np.newaxis]  # (N, 5, 1, 1)に拡張
            info_broadcasted = np.broadcast_to(info_expanded, (cwt_coefs.shape[0], info.shape[1], cwt_coefs.shape[2], cwt_coefs.shape[3]))
            cwt_coefs = np.concatenate((cwt_coefs, info_broadcasted), axis=1)

        if self.use_stance:
            start_col = self.signal_channels+5
            end_col = start_col+1

            info = X[:,0,start_col:end_col]
            info_expanded = info[:, :, np.newaxis, np.newaxis]  # (N, 5, 1, 1)に拡張
            info_broadcasted = np.broadcast_to(info_expanded, (cwt_coefs.shape[0], info.shape[1], cwt_coefs.shape[2], cwt_coefs.shape[3]))
            cwt_coefs = np.concatenate((cwt_coefs, info_broadcasted), axis=1)

        if self.use_count:
            start_col = self.signal_channels+6
            end_col = start_col+1

            info = X[:,0,start_col:end_col]/320
            info_expanded = info[:, :, np.newaxis, np.newaxis]  # (N, 5, 1, 1)に拡張
            info_broadcasted = np.broadcast_to(info_expanded, (cwt_coefs.shape[0], info.shape[1], cwt_coefs.shape[2], cwt_coefs.shape[3]))
            cwt_coefs = np.concatenate((cwt_coefs, info_broadcasted), axis=1)

        return cwt_coefs
    
    @staticmethod    
    def resize(images, height, width):
        resized=np.empty((images.shape[0],images.shape[1],height,width),dtype=np.float32)
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                resized[i,j]=cv2.resize(images[i,j], (width,height))

        return resized
    
def reverse_stance(data):
    '''
    チャンネルを反転させる
    data : (N,C,..)
    '''
    swaped = np.empty_like(data)
    swaped[:,::2], swaped[:,1::2] = data[:,1::2], data[:,::2]    
    return swaped

class SpectrogramScaler(BaseEstimator, TransformerMixin):
    def __init__(self, transform_type=None, normalize:str|float|int|None="max", image_layer_num=16, scale_maxmins=[]):
        '''
        transform_type : "log", "sqrt" or None
        normalize : max-min normalizationの最大値に何を使うか。"max":max, int:Nsigma, float:percentile(0-100)
        '''
        self.transform_type=transform_type
        self.normalize=normalize
        self.image_layer_num=image_layer_num
        self.scale_maxmins=scale_maxmins
        
    def fit(self, X, y=None):
        '''
        X : spectorogram images (N, C, H, W)
        '''
        if self.scale_maxmins == []:
            X = np.array(X)
            if self.transform_type == "log":
                X[:,:self.image_layer_num] = np.log(X[:,:self.image_layer_num])
            elif self.transform_type == "sqrt":
                X[:,:self.image_layer_num] = np.sqrt(X[:,:self.image_layer_num])

            self.scale_maxmins=[]
            if self.normalize is not None:
                for channel in range(self.image_layer_num):
                    Zxx_min = 0.0
                    Zxx_max = self.scaling_max(X[:,channel])
                    self.scale_maxmins.append({"min":Zxx_min,"max":Zxx_max})
        else:
            print("scale_maxmins are already set. fit is passed.")
        return self

    def transform(self, X):
        '''
        X : spectorogram images (N, C, H, W)
        '''
        X = np.array(X)
        if self.transform_type == "log":
            X[:,:self.image_layer_num] = np.log(X[:,:self.image_layer_num])
        elif self.transform_type == "sqrt":
            X[:,:self.image_layer_num] = np.sqrt(X[:,:self.image_layer_num])

        if self.normalize is not None:
            for channel in range(self.image_layer_num):
                Zxx_min = self.scale_maxmins[channel]["min"]
                Zxx_max = self.scale_maxmins[channel]["max"]

                X[:,channel] = (X[:,channel] - Zxx_min) / (Zxx_max - Zxx_min)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def scaling_max(self, arr):
        if self.normalize == "max":
            ret = np.max(arr)
        elif type(self.normalize)==int:
            ret = self.normalize*np.std(arr)
        elif type(self.normalize)==float:
            ret = np.percentile(arr, self.normalize)
        else:
            raise ValueError("'normalize' paramater must be float,int,None or 'max'")
        return ret