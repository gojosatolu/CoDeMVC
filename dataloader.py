from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py
import scipy.sparse

class NoisyMNIST(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'NoisyMNIST.mat')['trainLabel'].astype(np.int32).reshape(50000, ) - 1
        v1 = scipy.io.loadmat(path + 'NoisyMNIST.mat')['X1'].astype(np.float32)
        v2 = scipy.io.loadmat(path + 'NoisyMNIST.mat')['X2'].astype(np.float32)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(v1)
        self.V2 = scaler.fit_transform(v2)

    def __len__(self):
        return 50000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class COIL20(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'COIL20.mat')
        self.Y = data['gnd'].flatten().astype(np.int32) - 1
        v1 = data['fea'].astype(np.float32)
        # Construct view 2 with noise
        v2 = v1 + np.random.normal(0, 0.1, v1.shape).astype(np.float32)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(v1)
        self.V2 = scaler.fit_transform(v2)

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class ALOI(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'ALOI.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][3][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]), 
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class OutdoorScene(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'OutdoorScene.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][3][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]), 
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class YoutubeFace(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'YoutubeFace_sel_fea.mat')
        self.Y = data['Y'].flatten().astype(np.int32) - 1
        scaler = StandardScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][3][0].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X'][4][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]), 
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                torch.from_numpy(self.V5[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Caltech101_Test(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Caltech101.mat')
        self.Y = data['gt'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['fea'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['fea'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['fea'][0][2].astype(np.float32))
        self.V4 = scaler.fit_transform(data['fea'][0][3].astype(np.float32))
        self.V5 = scaler.fit_transform(data['fea'][0][4].astype(np.float32))
        self.V6 = scaler.fit_transform(data['fea'][0][5].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]), 
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                torch.from_numpy(self.V5[idx]), torch.from_numpy(self.V6[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class NoisyBDGP(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'noisy_BDGP.mat')
        self.Y = data['Y'].flatten().astype(np.int32)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X2'].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class NoisyHW(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'noisy_HW.mat')
        self.Y = data['Y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.V6 = scaler.fit_transform(data['X6'].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                torch.from_numpy(self.V5[idx]), torch.from_numpy(self.V6[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()



class MNIST_10k(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'mnist.mat')
        self.Y = data['truth'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        # Shape (1,2) -> (784, 10000), (256, 10000). Transpose required
        self.V1 = scaler.fit_transform(data['X'][0][0].T.astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].T.astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MSRC_v5(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'MSRC-v5.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        # Cell (5, 1) -> (210, D)
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][3][0].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X'][4][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
         return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                 torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                 torch.from_numpy(self.V5[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Reuters_1500(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Reuters-1500.mat')
        self.Y = data['y'].flatten().astype(np.int32) # 0-5, already 0-indexed presumably
        if self.Y.min() == 1: self.Y = self.Y - 1 # Safety
        scaler = MinMaxScaler()
        if scipy.sparse.issparse(data['X'][0][0]):
             self.V1 = scaler.fit_transform(data['X'][0][0].toarray().astype(np.float32))
        else:
             self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))

        if scipy.sparse.issparse(data['X'][1][0]):
             self.V2 = scaler.fit_transform(data['X'][1][0].toarray().astype(np.float32))
        else:
             self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))

        if scipy.sparse.issparse(data['X'][2][0]):
             self.V3 = scaler.fit_transform(data['X'][2][0].toarray().astype(np.float32))
        else:
             self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))

        if scipy.sparse.issparse(data['X'][3][0]):
             self.V4 = scaler.fit_transform(data['X'][3][0].toarray().astype(np.float32))
        else:
             self.V4 = scaler.fit_transform(data['X'][3][0].astype(np.float32))

        if scipy.sparse.issparse(data['X'][4][0]):
             self.V5 = scaler.fit_transform(data['X'][4][0].toarray().astype(np.float32))
        else:
             self.V5 = scaler.fit_transform(data['X'][4][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
         return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                 torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                 torch.from_numpy(self.V5[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class UCI(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'UCI.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        # Cell (3, 1) -> (2000, D)
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class FashionMNIST(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'fmnist.mat')
        # Labels in 'truelabel' (3, 1)? Likely one array inside or replicated?
        # Checked truelabel shape (3,1)? Wait, fmnist 60000 samples. 
        # Usually truelabel corresponds. 
        # Inspecting truelabel array content might be necessary but let's assume it's like others or data['truelabel'][0][0]
        # data['data']: (3,1) cell containing (512, 60000)
        # Need Transpose
        
        # Label handling logic requires care. Assuming unique labels
        # Let's try to extract Y from one of the cells if truelabel is weird
        # BUT: fmnist.mat usually has correct labels. 
        # truelabel shape (3,1) is suspicious, maybe cell array of labels for each view?
        # We'll take the first one.
        
        lab_ref = data['truelabel']
        if lab_ref.shape == (3, 1): # If cell array
             self.Y = lab_ref[0][0].flatten().astype(np.int32)
        else:
             self.Y = lab_ref.flatten().astype(np.int32)
        if self.Y.min() == 1: self.Y = self.Y - 1

        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['data'][0][0].T.astype(np.float32))
        self.V2 = scaler.fit_transform(data['data'][1][0].T.astype(np.float32))
        self.V3 = scaler.fit_transform(data['data'][2][0].T.astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class NUSWIDE_OBJ(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'NUSWIDEOBJ.mat')
        self.Y = data['Y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        # Cell (5, 1) -> (30000, D)
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][3][0].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X'][4][0].astype(np.float32))
        
        # Sparse check for NUS features
        if scipy.sparse.issparse(self.V1): self.V1 = self.V1.toarray()
        if scipy.sparse.issparse(self.V2): self.V2 = self.V2.toarray()
        if scipy.sparse.issparse(self.V3): self.V3 = self.V3.toarray()
        if scipy.sparse.issparse(self.V4): self.V4 = self.V4.toarray()
        if scipy.sparse.issparse(self.V5): self.V5 = self.V5.toarray()

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
         return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                 torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                 torch.from_numpy(self.V5[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class ORL(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'ORL.mat')
        self.Y = data['gt'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        # Cell (1, 3) -> (400, D)
        self.V1 = scaler.fit_transform(data['fea'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['fea'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['fea'][0][2].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
         return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                 torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class EYaleB(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'EYaleB10_mtv.mat')
        self.Y = data['gt'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        # Cell (1, 3) -> (640, D)
        self.V1 = scaler.fit_transform(data['fea'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['fea'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['fea'][0][2].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
         return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                 torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class TinyImage(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'tinyimage.mat')
        self.Y = data['truelabel'][0][0].flatten().astype(np.int32) - 1
        scaler = StandardScaler()
        # Transpose needed: (Features, Samples) -> (Samples, Features)
        self.V1 = scaler.fit_transform(data['data'][0][0].T.astype(np.float32))
        self.V2 = scaler.fit_transform(data['data'][1][0].T.astype(np.float32))
        self.V3 = scaler.fit_transform(data['data'][2][0].T.astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class STL10(Dataset):
    def __init__(self, path):
        f = h5py.File(path + 'stl10_fea.mat', 'r')
        self.Y = np.array(f['Y']).flatten().astype(np.int32) - 1
        # h5py loads as (Features, Samples) usually for MATLAB v7.3
        # We need to dereference the object references
        # X is (3, 1)
        ref1 = f['X'][0][0]
        ref2 = f['X'][1][0]
        ref3 = f['X'][2][0]
        
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(np.array(f[ref1]).T.astype(np.float32))
        self.V2 = scaler.fit_transform(np.array(f[ref2]).T.astype(np.float32))
        self.V3 = scaler.fit_transform(np.array(f[ref3]).T.astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MSRC_v1(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'MSRC_v1.mat')
        self.Y = data['truth'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['msr1'].astype(np.float32))
        self.V2 = scaler.fit_transform(data['msr2'].astype(np.float32))
        self.V3 = scaler.fit_transform(data['msr3'].astype(np.float32))
        self.V4 = scaler.fit_transform(data['msr4'].astype(np.float32))
        self.V5 = scaler.fit_transform(data['msr5'].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                torch.from_numpy(self.V5[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class NUS_WIDE(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'NUS-WIDE.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        # Mat shape is (5, 1) cell array, accessed as data['X'][i][0]
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][3][0].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X'][4][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                torch.from_numpy(self.V5[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Caltech101_20(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Caltech101-20.mat')
        self.Y = data['truth'].flatten().astype(np.int32) # Original likely 0-indexed or handled elsewhere if 1-indexed, but -1 caused min to be -1
        if self.Y.min() == 1:
             self.Y = self.Y - 1
        scaler = MinMaxScaler()
        # Shape: [(48, 2386), (40, 2386), (254, 2386), (1984, 2386), (512, 2386), (928, 2386)]
        # Need Transpose
        self.V1 = scaler.fit_transform(data['X'][0][0].T.astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].T.astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][0][2].T.astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][0][3].T.astype(np.float32))
        self.V5 = scaler.fit_transform(data['X'][0][4].T.astype(np.float32))
        self.V6 = scaler.fit_transform(data['X'][0][5].T.astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                torch.from_numpy(self.V5[idx]), torch.from_numpy(self.V6[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Animal(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Animal.mat')
        self.Y = data['Y'].flatten().astype(np.int32) - 1  # 1-indexed to 0-indexed
        # Animal has 4 views with very different scales
        # View 0: [0, 1] already normalized
        # View 1: [0, 12694] needs scaling  
        # View 2: [0, 2156] needs scaling
        # View 3: [0, 1] already normalized
        # Use MinMaxScaler for consistency across all views
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][0][3].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class cifar_10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class cifar_100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)],self.Y[idx], torch.from_numpy(np.array(idx)).long()

class synthetic3d():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'synthetic3d.mat')
        self.Y = data['Y'].astype(np.int32).reshape(600,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
               self.Y[idx], torch.from_numpy(np.array(idx)).long()

class prokaryotic():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32).reshape(551,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 551
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
               self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Reuters_1200(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Reuters-1200.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MaxAbsScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][3][0].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X'][4][0].astype(np.float32))
        
        if scipy.sparse.issparse(self.V1): self.V1 = self.V1.toarray()
        if scipy.sparse.issparse(self.V2): self.V2 = self.V2.toarray()
        if scipy.sparse.issparse(self.V3): self.V3 = self.V3.toarray()
        if scipy.sparse.issparse(self.V4): self.V4 = self.V4.toarray()
        if scipy.sparse.issparse(self.V5): self.V5 = self.V5.toarray()

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                torch.from_numpy(self.V5[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class NottingHill(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Notting-Hill.mat')
        self.Y = data['gt'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        # Cell (1, 3) -> (550, D)
        self.V1 = scaler.fit_transform(data['fea'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['fea'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['fea'][0][2].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class YaleB(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'YaleB_32x32.mat')
        self.Y = data['gnd'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        v1 = data['fea'].astype(np.float32)
        v2 = v1 + np.random.normal(0, 0.1, v1.shape).astype(np.float32)
        self.V1 = scaler.fit_transform(v1)
        self.V2 = scaler.fit_transform(v2)

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class BBCSport(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'bbcsport.mat')
        self.Y = data['Y'].flatten().astype(np.int32) - 1
        scaler = MaxAbsScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        
        if scipy.sparse.issparse(self.V1): self.V1 = self.V1.toarray()
        if scipy.sparse.issparse(self.V2): self.V2 = self.V2.toarray()

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class LandUse_21(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'LandUse-21.mat')
        self.Y = data['Y'].flatten().astype(np.int32) - 1
        scaler = MaxAbsScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        
        if scipy.sparse.issparse(self.V1): self.V1 = self.V1.toarray()
        if scipy.sparse.issparse(self.V2): self.V2 = self.V2.toarray()
        if scipy.sparse.issparse(self.V3): self.V3 = self.V3.toarray()

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Scene_15(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Scene-15.mat')
        self.Y = data['Y'].flatten().astype(np.int32) - 1
        scaler = MaxAbsScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        
        if scipy.sparse.issparse(self.V1): self.V1 = self.V1.toarray()
        if scipy.sparse.issparse(self.V2): self.V2 = self.V2.toarray()
        if scipy.sparse.issparse(self.V3): self.V3 = self.V3.toarray()

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class WebKB(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'WebKB.mat')
        self.Y = data['gnd'].flatten().astype(np.int32) - 1
        scaler = MaxAbsScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        
        if scipy.sparse.issparse(self.V1): self.V1 = self.V1.toarray()
        if scipy.sparse.issparse(self.V2): self.V2 = self.V2.toarray()

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class STL10_4V(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'STL10_4views.mat')
        self.Y = data['Y'].flatten().astype(np.int32)
        
        if self.Y.min() == 1:
            self.Y = self.Y - 1

        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X4'].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), 
                torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx]),
                torch.from_numpy(self.V4[idx])], \
               self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Yale(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Yale.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        # Cell array (V, 1), accessed as [v][0]
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
         return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                 torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class OneHundredLeaves(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + '100Leaves.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Handwritten(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'handwritten.mat')
        self.Y = data['Y'].flatten().astype(np.int32)
        # Already 0-indexed 0-9
        scaler = MinMaxScaler()
        # Cell array (1, V), accessed as [0][v]
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][0][3].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X'][0][4].astype(np.float32))
        self.V6 = scaler.fit_transform(data['X'][0][5].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]),
                torch.from_numpy(self.V5[idx]), torch.from_numpy(self.V6[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class ThreeSources(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + '3Sources.mat')
        self.Y = data['y'].flatten().astype(np.int32) - 1
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))

    def __len__(self):
        return self.V1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),
                torch.from_numpy(self.V3[idx])], self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "Synthetic3d":
        dataset = synthetic3d('./data/')
        dims = [3,3,3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == "Prokaryotic":
        dataset = prokaryotic('./data/')
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4
    elif dataset == "Cifar10":
        dataset = cifar_10('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
    elif dataset == "Cifar100":
        dataset = cifar_100('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100
    elif dataset == "NoisyMNIST":
        dataset = NoisyMNIST('./data/')
        dims = [784, 784]
        view = 2
        data_size = 50000
        class_num = 10
    elif dataset == "COIL20":
        dataset = COIL20('./data/')
        dims = [1024, 1024]
        view = 2
        data_size = 1440
        class_num = 20
    elif dataset == "ALOI":
        dataset = ALOI('./data/')
        dims = [77, 13, 64, 125]
        view = 4
        data_size = 10800
        class_num = 100
    elif dataset == "OutdoorScene":
        dataset = OutdoorScene('./data/')
        dims = [512, 432, 256, 48]
        view = 4
        data_size = 2688
        class_num = 8
    elif dataset == "YoutubeFace":
        dataset = YoutubeFace('./data/')
        dims = [64, 512, 64, 647, 838]
        view = 5
        data_size = 101499
        class_num = 31
    elif dataset == "Caltech101":
        dataset = Caltech101_Test('./data/')
        dims = [48, 40, 254, 1984, 512, 928]
        view = 6
        data_size = 9144
        class_num = 102
    elif dataset == "NoisyBDGP":
        dataset = NoisyBDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "NoisyHW":
        dataset = NoisyHW('./data/')
        dims = [216, 76, 64, 6, 240, 47]
        view = 6
        data_size = 2000
        class_num = 10
    elif dataset == "TinyImage":
        dataset = TinyImage('./data/')
        dims = [512, 512, 1280]
        view = 3
        data_size = 100000
        class_num = 200
    elif dataset == "STL10":
        dataset = STL10('./data/')
        dims = [1024, 512, 2048]
        view = 3
        data_size = 13000
        class_num = 10
    elif dataset == "MSRC-v1":
        dataset = MSRC_v1('./data/')
        dims = [24, 576, 512, 256, 254]
        view = 5
        data_size = 210
        class_num = 7  # MSRC typically has 7 classes for v1
    elif dataset == "NUS-WIDE":
        dataset = NUS_WIDE('./data/')
        # Shapes: [(2400, 64), (2400, 144), (2400, 73), (2400, 128), (2400, 225)]
        dims = [64, 144, 73, 128, 225]
        view = 5
        data_size = 2400
        class_num = 31 # NUS-WIDE-OBJ typically 31
    elif dataset == "Caltech101-20":
        dataset = Caltech101_20('./data/')
        # Shape: [(48, 2386), (40, 2386), (254, 2386), (1984, 2386), (512, 2386), (928, 2386)]
        dims = [48, 40, 254, 1984, 512, 928]
        view = 6
        data_size = 2386
        class_num = 20
    elif dataset == "Animal":
        dataset = Animal('./data/')
        # Shape: [(11673, 2689), (11673, 2000), (11673, 2001), (11673, 2000)]
        dims = [2689, 2000, 2001, 2000]
        view = 4
        data_size = 11673
        class_num = 20
    elif dataset == "Reuters-1200":
        dataset = Reuters_1200('./data/')
        dims = [2000, 2000, 2000, 2000, 2000]
        view = 5
        data_size = 1200
        class_num = 6
    elif dataset == "BBCSport":
        dataset = BBCSport('./data/')
        dims = [3183, 3203]
        view = 2
        data_size = 544
        class_num = 5
    elif dataset == "LandUse-21":
        dataset = LandUse_21('./data/')
        dims = [20, 59, 40]
        view = 3
        data_size = 2100
        class_num = 21
    elif dataset == "Scene-15":
        dataset = Scene_15('./data/')
        dims = [20, 59, 40]
        view = 3
        data_size = 4485
        class_num = 15
    elif dataset == "WebKB":
        dataset = WebKB('./data/')
        dims = [2949, 334]
        view = 2
        data_size = 1051
        class_num = 2
    elif dataset == "STL10_4V":
        dataset = STL10_4V('./data/')
        dims = [512, 2048, 4096, 1024] 
        view = 4
        data_size = 13000
        class_num = 10
    elif dataset == "Yale":
        dataset = Yale('./data/')
        dims = [4096, 3304, 6750]
        view = 3
        data_size = 165
        class_num = 15
    elif dataset == "100Leaves":
        dataset = OneHundredLeaves('./data/')
        dims = [64, 64, 64]
        view = 3
        data_size = 1600
        class_num = 100
    elif dataset == "Handwritten":
        dataset = Handwritten('./data/')
        dims = [240, 76, 216, 47, 64, 6]
        view = 6
        data_size = 2000
        class_num = 10
    elif dataset == "3Sources":
        dataset = ThreeSources('./data/')
        dims = [3560, 3631, 3068]
        view = 3
        data_size = 169
        class_num = 6
    elif dataset == "MNIST-10k":
        dataset = MNIST_10k('./data/')
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "MSRC-v5":
        dataset = MSRC_v5('./data/')
        dims = [24, 576, 512, 256, 254]
        view = 5
        data_size = 210
        class_num = 7
    elif dataset == "Reuters-1500":
        dataset = Reuters_1500('./data/')
        # Dims confirmed: [21531, 24892, 34251, 15506, 11547]
        dims = [21531, 24892, 34251, 15506, 11547]
        view = 5
        data_size = 1500
        class_num = 6 # 0-5
    elif dataset == "UCI":
        dataset = UCI('./data/')
        dims = [240, 76, 6]
        view = 3
        data_size = 2000
        class_num = 10
    elif dataset == "FashionMNIST":
        # Dims: (512, 60000), (512, 60000), (1280, 60000). Transposed -> 512, 512, 1280
        dataset = FashionMNIST('./data/') 
        dims = [512, 512, 1280] 
        view = 3
        data_size = 60000
        class_num = 10 
    elif dataset == "NUS-WIDE-OBJ":
        dataset = NUSWIDE_OBJ('./data/')
        dims = [65, 226, 145, 74, 129]
        view = 5
        data_size = 30000
        class_num = 31
    elif dataset == "ORL":
        dataset = ORL('./data/')
        # Dims: 4096, 3304, 6750
        dims = [4096, 3304, 6750]
        view = 3
        data_size = 400
        class_num = 40
    elif dataset == "EYaleB":
        dataset = EYaleB('./data/')
        # Dims: 1024, 1239, 256
        dims = [1024, 1239, 256]
        view = 3
        data_size = 640
        class_num = 10
    elif dataset == "NottingHill":
        dataset = NottingHill('./data/')
        # Dims confirmed: 2000, 3304, 6750
        dims = [2000, 3304, 6750] 
        view = 3
        data_size = 550
        class_num = 5
    elif dataset == "YaleB":
        dataset = YaleB('./data/')
        # V1: 1024, V2: 1024
        dims = [1024, 1024]
        view = 2
        data_size = 2414
        class_num = 38
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
