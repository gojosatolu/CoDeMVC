from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py

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
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
