from copy import deepcopy
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import timm

from emg_utils import *

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True

class Timm112(nn.Module):
    def __init__(self, model_name:str, signal_channels=16, output_shape=(30,3), subject_num=5, activate_params=None):
        super().__init__()

        self.signal_channels = signal_channels
        self.output_shape = output_shape
        self.activate_params = activate_params

        # self.upsample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.backborn = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=self.signal_channels,
            num_classes=0,
        )

        num_features = self.backborn.num_features
        self.fc_out = nn.Linear(num_features, output_shape[0]*output_shape[1])
        self.fc_label = nn.Linear(num_features, subject_num)
        

    def forward(self, x):
        x = self.backborn(x)
        out = self.fc_out(x)
        out = out.view(-1, self.output_shape[0], self.output_shape[1])

        label = self.fc_label(x)

        if self.activate_params:
            a,b,c=self.activate_params
            out[:,:,2] = a*torch.sin(out[:,:,0]/b)+c+torch.tanh(out[:,:,2])
        return out, label
    
    def renew_first_conv(self,**kwargs):
        name,initial = self.get_first_conv()
        initial_param = dict(
            in_channels=initial.in_channels,
            out_channels=initial.out_channels,
            kernel_size=initial.kernel_size,
            stride=initial.stride,
            padding=initial.padding,
            dilation=initial.dilation,
            groups=initial.groups,
            bias=initial.bias is not None,
            padding_mode=initial.padding_mode,
        )
        new_param = {**initial_param, **kwargs}
        new_layer = nn.Conv2d(**new_param)
        # self.copy_param_if_possible(initial, new_layer)
        self.set_module_by_name(self.backborn, name, new_layer) 
        
    def get_first_conv(self):
        config=self.backborn.default_cfg
        first_conv_name = config["first_conv"]
        first_conv_layer = dict(self.backborn.named_modules())[first_conv_name]
        return first_conv_name, first_conv_layer

    @staticmethod    
    def set_module_by_name(model, module_name, new_module):
        parent_module = model
        name_parts = module_name.split('.')
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name_parts[-1], new_module)

    @staticmethod
    def copy_param_if_possible(src, dest):
        src_dict = src.state_dict()
        dest_dict = dest.state_dict()
        if src_dict["weight"].shape==dest_dict["weight"].shape:
            dest.load_state_dict(src_dict)

class CogDataset(Dataset):
    def __init__(self, signals, motion, signal_channels=16, additional_channels=0, subject_num=5, device="cpu", transform=None, mirror_augmentation=None, time_padding=0, freq_padding=0, specific_subject=None):
        '''
        signals : (N, signal_channels+subject_num, H, W)
        motion : (N, T, XYZ)
        specific_subject : 0001 - 0004
        '''
        subject_onehot = signals[:,-subject_num:,0,0]
        subject_labels = np.argmax(subject_onehot, axis=1)

        if specific_subject is not None:
            # subject_labelsに一致するデータのみにする。
            specific_label = int(specific_subject)-1
            mask = subject_labels == specific_label 
            
            signals = signals[mask]
            motion = motion[mask]
            
        self.signals = torch.tensor(signals[:,:signal_channels+additional_channels],dtype=torch.float32,device=device)
        self.subject = torch.tensor(subject_labels,device=device)
        self.target = torch.tensor(motion,dtype=torch.float32,device=device)

        if time_padding or freq_padding:
            self.signals = torch.nn.functional.pad(self.signals, (time_padding,time_padding,freq_padding,freq_padding), mode='constant')

        self.signal_channels = signal_channels
        self.additional_channels = additional_channels
        self.subject_num = subject_num

        self.transform = transform
        self.mirror_augmentation = mirror_augmentation

        self.height = self.signals.size(2)
        self.width = self.signals.size(3)

    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        item = {
            'signals': self.signals[idx],
            'target': self.target[idx],
            'subject': self.subject[idx],
            'index': idx
        }

        output_size=112
        if self.height > output_size or self.width > output_size:
            if self.height % output_size == 0 and self.width % output_size == 0:
                # 整数倍の時はステップで抽出
                item["signals"] = self._random_slice(item["signals"], output_size)
            else:
                # 112より大きい場合はランダムクロップ
                item["signals"],_ = self._random_crop(item["signals"], output_size)

        if self.transform:
            item["signals"] = self.transform(item["signals"])

        if self.mirror_augmentation is not None and self.mirror_augmentation[idx] and np.random.rand() < 0.5:
            item["signals"] = self._mirror_signals(item["signals"])
            # item["target"] = self._mirror_motion(item["target"]) #スタンスに関わらずXYの動きは同一
            item["subject"] = self.subject[idx]+self.subject_num


        return item
    
    def _random_crop(self, image, size):
        '''
        sizeにランダムクロップ
        '''
        start_idx_h = np.random.randint(0, self.height - size + 1) if self.height > size else 0
        start_idx_w = np.random.randint(0, self.width - size + 1) if self.width > size else 0
        image = image[:, start_idx_h:start_idx_h + size, start_idx_w:start_idx_w + size]
        return image,(start_idx_h,start_idx_w)
    
    def _random_slice(self, image, size):
        '''
        sizeにスライス
        '''
        step_h = self.height // size
        step_w = self.width // size
        start_idx_h = np.random.randint(0, step_h)
        start_idx_w = np.random.randint(0, step_w)
        image = image[:, start_idx_h::step_h, start_idx_w::step_w]
        return image
    
    def _shift_target(self, target, time_shift):
        '''
        targetをシフトする
        シフトが1/30以上なら、1データずらす
        padding7以下なら2データずらすことはない
        '''
        if time_shift > 1/30:
            shifted_target = torch.zeros_like(target)
            shifted_target[1:]=target[:-1]
            shifted_target[0]=2*shifted_target[1]-shifted_target[2] #線形で外挿
            return shifted_target

        elif time_shift < -1/30:
            shifted_target = torch.zeros_like(target)
            shifted_target[:-1]=target[1:]
            shifted_target[-1]=2*shifted_target[-2]-shifted_target[-3] #線形で外挿
            return shifted_target

        else:
            return target
    
    def _mirror_signals(self, signals):
        '''
        チャンネルの偶数奇数を入れ替える
        signals: (C, ...)
        mode : 0  オーグメンテーション
               1  左右入れ替え
        '''
        mirrored_signals = signals.clone()
        mirrored_signals[:self.signal_channels:2], mirrored_signals[1:self.signal_channels:2] = signals[1:self.signal_channels:2], signals[:self.signal_channels:2]
        return mirrored_signals
    
    def _mirror_motion(self, motion):
        '''
        Yを反転する
        motion: (T, XYZ)
        '''
        mirrored_motion = motion.clone()
        mirrored_motion[:,1] = -motion[:,1]
        return mirrored_motion
    
class ChannelRandomErasing:
    def __init__(self, **kwargs):
        self.eraser = v2.RandomErasing(**kwargs)

    def __call__(self, img):
        img_copy = img.clone()
        for ch in range(img.shape[0]):
            img_copy[ch] = self.eraser(img[ch])
        return img_copy
    
class GaussianNoise:
    def __init__(self, std=0.02):
        self.std=std

    def __call__(self, data):
        if self.std:
            noise = torch.normal(1, self.std, size=data.shape, device=data.device)
            data_noisy = data * noise
            return data_noisy
        else:
            return data
    
class EMGModel():
    def __init__(self, num_epochs, batch_size=64, lr=0.001, weight_decay=1e-4, momentum=0.0,
                 p_width=0.0, p_channel=0.0, noise_level=0.0, train_mirror_data=False, time_padding=0, freq_padding=0,
                 model_name = None,
                 torch_model=Timm112, model_state=None, unfreeze_epoch=None,
                 model_history_len=1,
                 additional_channels=0,
                 signal_channels=16,
                 output_shape=(30,3),
                 dropout=0.1,
                 loss_ratio=[0.99,0.01],
                 xyz_weight=[1.,1.,1.],
                 train_subject:str="",
                 switch_epochs=-1,
                 lr_scheduler=None,
                 seed=1,
                 penalty_params=None,
                 activate_params=None,
                ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        self.seed = seed
        self.model = None
        self.signal_channels = signal_channels
        self.additional_channles = additional_channels
        self.time_padding = time_padding
        self.freq_padding = freq_padding

        self.p_width = p_width
        self.p_channel = p_channel
        self.train_mirror_data = train_mirror_data

        self.model_name = model_name
        self.torch_model = torch_model
        self.model_state = model_state #事前学習結果
        self.unfreeze_epoch = unfreeze_epoch #フリーズ解除するepoch (モデルにfreeze_pretrainedメソッドが必須)

        self.model_history_len = model_history_len #最後からこのepoch数だけモデルを保存する
        self.model_history = []
        self.minimum_loss = 1e10
        self.minimum_loss_weight=None

        self.output_shape = output_shape
        self.dropout = dropout

        self.train_subject=train_subject #途中から特定のsubjectのみで学習する場合
        self.switch_epochs = switch_epochs if switch_epochs>=0 else self.num_epochs//2 #特定のsubjectに切り替えるエポック数
        self.lr_scheduler = lr_scheduler #classとparamのtupple
        self.loss_ratio = loss_ratio #マルチタスク学習時のlossの主タスクの割合
        self.xyz_weight = xyz_weight
        self.penalty_params = penalty_params
        self.activate_params = activate_params

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fix_seed(seed)

        self.train_transforms=None
        self.train_transforms=v2.Compose([
            GaussianNoise(std=noise_level),
        ])

        self.initialize_model()

    @staticmethod
    def fix_seed(seed):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    def initialize_model(self):
        if self.model_name:
            self.model = Timm112(self.model_name, signal_channels=self.signal_channels, subject_num=5 if not self.train_mirror_data else 10, activate_params=self.activate_params)
        else:
            self.model = self.torch_model()

        if self.model_state is not None:
            self.model.load_state_dict(self.model_state)

        # 途中から学習する層がある場合は、特定のパラメータを固定する
        if self.unfreeze_epoch is not None:
            unfreeze_params(self.model)
            self.model.freeze_pretrained()

    def fit(self, in_images, out_arr, verbose=0, val_X=None,val_y=None,mirror_augmentation=None):
        '''
        速度とsubjectを同時に学習する
        エポックの途中で学習データを特定subjectのみに切り替える
        in_images : ndarray(N, C, H, W)
        out_arr : ndarray(N, t, C)
        '''
        self.model.to(self.device)
        if hasattr(self.model, "create_optimizer"):
            optimizer = self.model.create_optimizer(lr=self.lr, weight_decay=self.weight_decay)
        else:
            # optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            # optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.lr_scheduler:
            scheduler = self.lr_scheduler[0](optimizer, **self.lr_scheduler[1])

        criterion = rmse3D
        criterion_label = nn.CrossEntropyLoss()

        dataset = CogDataset(in_images, out_arr, signal_channels=self.signal_channels,transform=self.train_transforms, mirror_augmentation=mirror_augmentation, device="cuda", additional_channels=self.additional_channles, time_padding=self.time_padding,freq_padding=self.freq_padding)
        sub_dataset = CogDataset(in_images, out_arr, signal_channels=self.signal_channels,transform=self.train_transforms, mirror_augmentation=None, specific_subject=self.train_subject, device="cuda", additional_channels=self.additional_channles, time_padding=self.time_padding,freq_padding=self.freq_padding)
        g = torch.Generator()
        g.manual_seed(self.seed)
        if isinstance(self.batch_size, (list, tuple)):
            first_batch_size,second_batch_size=self.batch_size[:2]
            # 小数であればデータセットの倍数
            if type(second_batch_size)==float:
                second_batch_size=math.ceil(len(sub_dataset)*second_batch_size)
        else:
            first_batch_size=self.batch_size
            second_batch_size=self.batch_size
        
        train_loader = DataLoader(dataset, batch_size=first_batch_size, shuffle=True, drop_last=False, num_workers=0)
        sub_train_loader = DataLoader(sub_dataset, batch_size=second_batch_size, shuffle=True, num_workers=0)
        sub_val_loader = None

        if val_X is not None:
            assert val_X.shape[3]==112
            val_dataset = CogDataset(val_X, val_y, signal_channels=self.signal_channels, transform=None, mirror_augmentation=None, specific_subject=self.train_subject, device="cuda",additional_channels=self.additional_channles)
            sub_val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        train_loss_history=[]
        label_loss_history=[]
        val_loss_history=[]
        for epoch in range(self.num_epochs):
            self.model.train()
            if self.unfreeze_epoch is not None and epoch==self.unfreeze_epoch:
                unfreeze_params(self.model)

            train_loss = 0.0
            label_loss = 0.0
            if epoch < self.switch_epochs:
                loader=train_loader
            else:
                loader=sub_train_loader

            for batch in loader:
                images, targets, label = batch["signals"].to(self.device), batch["target"].to(self.device), batch["subject"].to(self.device)
                # print(batch["signals"][0,0,0,0],batch["index"][0])

                optimizer.zero_grad()
                outputs, label_outputs = self.model(images)

                loss1 = criterion(outputs, targets, weight=self.xyz_weight)
                loss2 = criterion_label(label_outputs, label)
                loss = self.loss_ratio[0]*loss1+ self.loss_ratio[1]*loss2
                loss.backward()
                optimizer.step()
                train_loss += loss1.item()
                label_loss += loss2.item()

                # print(batch["index"][:5])

            if self.lr_scheduler:
                scheduler.step()

            train_loss = train_loss / len(loader.dataset)
            label_loss = label_loss / len(loader.dataset)
            train_loss_history.append(train_loss)
            label_loss_history.append(label_loss)

            if sub_val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in sub_val_loader:
                        images, targets, label = batch["signals"], batch["target"], batch["subject"]
                        val_outputs,_ = self.model(images)
                        val_loss += rmse3D(targets, val_outputs).item()
                val_loss /= len(sub_val_loader.dataset)
                val_loss_history.append(val_loss)

                if verbose or epoch+1==self.num_epochs:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {train_loss:.4f}, Labelloss: {label_loss:.4g}, Val Loss: {val_loss:.4f}')
            else:
                if verbose or epoch+1==self.num_epochs:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {train_loss:.4f}, Labelloss: {label_loss:.4g}')

            self.save_model_history(self.model, val_loss if sub_val_loader else None)

        train_loss_history=np.array(train_loss_history)
        label_loss_history=np.array(label_loss_history)
        val_loss_history=np.array(val_loss_history)
        
        return train_loss_history, val_loss_history, label_loss_history

    def predict(self, X:np.ndarray, transform=False, history="mean")->np.ndarray:
        '''
        prediction for sklearn
        X : scaled spectrum images ndarray(N, C, H, W)
        history : which weight used. "mean","last" or "best"
        '''
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.to(self.device)
        self.model.eval()
        pred_list=[]

        with torch.no_grad():
            if history=="best":
                self.model.load_state_dict(self.minimum_loss_weight)

                pred,_ = self.model(X_tensor[:,:self.signal_channels])
                pred_list.append(pred)
            elif history=="last":
                self.model.load_state_dict(self.model_history[-1])

                pred,_ = self.model(X_tensor[:,:self.signal_channels])
                pred_list.append(pred)
            elif history=="mean":
                for state in self.model_history:
                    self.model.load_state_dict(state)

                    pred,_ = self.model(X_tensor[:,:self.signal_channels])
                    pred_list.append(pred)
            else:
                raise ValueError(f"{history=}")

        predictions = torch.mean(torch.stack(pred_list), axis=0)
        ret = self.postprocess(predictions, transform)
        return ret
    
    def postprocess(self, cnn_output, transform):
        '''
        cnn_output : tensor(N, 30, 3)
        return : numpy(N, 30, 3)
        '''
        ret=cnn_output.cpu().numpy()
        if transform:
            ret=transform_vel(ret)

        return ret

    def save_model_history(self,model:nn.Module, loss=None):
        '''
        save snapshot
        '''
        model.eval()
        self.model_history.append(deepcopy(model.state_dict()))
        self.model_history=self.model_history[-self.model_history_len:]

        if loss is not None and loss < self.minimum_loss:
            self.minimum_loss=loss
            self.minimum_loss_weight=deepcopy(model.state_dict())
        