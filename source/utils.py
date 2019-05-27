from PIL import Image
import torchvision.transforms as t
import torch.utils.data as data

def load_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

train_trans = t.Compose([
    t.Resize((256, 256)),
    t.RandomCrop(224),
    t.RandomHorizontalFlip(),
    t.ToTensor(),
    t.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

test_trans = t.Compose([
    t.Resize((224, 224)),
    t.ToTensor(),
    t.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])



class ImageDataset_2015(data.Dataset):
    def __init__(self, train=True):
       
        dataset_path = '../dataset/2015/'

        with open(dataset_path + 'Train.csv') as f:
            train_file = f.readlines()
        with open(dataset_path + 'Validation.csv') as f:
            val_file = f.readlines()
        with open(dataset_path + 'Test.csv') as f:
            test_file = f.readlines()

        self.train = train
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        test_labels2 = []

        datatype_path = 'Faces'

        for item in train_file:
            r = item.split(';')
            train_data.append(dataset_path + 'Train' + datatype_path + '/%s' % r[0])
            train_labels.append(r[1])

        for item in val_file:
            r = item.split(';')
            train_data.append(dataset_path + 'Validation' + datatype_path + '/%s' % r[0])
            train_labels.append(r[1])

        for item in test_file:
            r = item.split(';')
            test_data.append(dataset_path + 'Test' + datatype_path + '/%s' % r[0])
            test_labels.append(r[1])
            test_labels2.append(r[2][:-1])

        self.data = None
        self.labels = None
        if self.train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels
            self.labels2 = test_labels2

            
    def __getitem__(self, index):
        img_path = self.data[index]
        img = load_image(img_path)
        if self.train:
            img = train_trans(img)
        else:
            img = test_trans(img)

        label = int(self.labels[index])
        class_label = label

        if self.train:
            return img, label, class_label
        else:
            return img, label, float(self.labels2[index])

    def __len__(self):
        return len(self.data)

class ImageDataset_2016(data.Dataset):
    def __init__(self, train=True):
       
        dataset_path = '../dataset/2016/'

        with open(dataset_path + 'Train.csv') as f:
            train_file = f.readlines()
        with open(dataset_path + 'Validation.csv') as f:
            val_file = f.readlines()
        with open(dataset_path + 'Test.csv') as f:
            test_file = f.readlines()

        self.train = train
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        test_labels2 = []

        datatype_path = 'Faces'

        for item in train_file:
            r = item.split(',')
            train_data.append(dataset_path + 'Train' + datatype_path + '/%s' % r[0])
            train_labels.append(r[1])

        for item in val_file:
            r = item.split(',')
            train_data.append(dataset_path + 'Validation' + datatype_path + '/%s' % r[0])
            train_labels.append(r[1])

        for item in test_file:
            r = item.split(',')
            test_data.append(dataset_path + 'Test' + datatype_path + '/%s' % r[0])
            test_labels.append(r[1])
            test_labels2.append(r[2][:-1])

        self.data = None
        self.labels = None
        if self.train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels
            self.labels2 = test_labels2

            
    def __getitem__(self, index):
        img_path = self.data[index]
        img = load_image(img_path)
        if self.train:
            img = train_trans(img)
        else:
            img = test_trans(img)

        label = float(self.labels[index])
        class_label = round(label)

        if self.train:
            return img, label, class_label
        else:
            return img, label, float(self.labels2[index])

    def __len__(self):
        return len(self.data)

class ImageDataset_wiki(data.Dataset):
    def __init__(self, train=True):
       
        dataset_path = '../dataset/wiki/'

        with open(dataset_path + 'Train.csv') as f:
            train_file = f.readlines()
        with open(dataset_path + 'Test.csv') as f:
            test_file = f.readlines()

        self.train = train
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        datatype_path = 'faces/'
        
        for item in train_file:
            r = item.split(',')
            train_data.append(dataset_path + datatype_path + '/%s' % r[0])
            train_labels.append(r[1][:-1])

        for item in test_file:
            r = item.split(',')
            test_data.append(dataset_path + datatype_path + '/%s' % r[0])
            test_labels.append(r[1][:-1])

        self.data = None
        self.labels = None
        if self.train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels
            
    def __getitem__(self, index):
        img_path = self.data[index]
        img = load_image(img_path)
        if self.train:
            img = train_trans(img)
        else:
            img = test_trans(img)

        label = int(self.labels[index])
        class_label = label

        if self.train:
            return img, label, class_label
        else:
            return img, label, 1

    def __len__(self):
        return len(self.data)


class ImageDataset_imdb_wiki(data.Dataset):
    def __init__(self, train=True):
       
        dataset_path = '../dataset/wiki/'

        with open(dataset_path + 'Train.csv') as f:
            train_file = f.readlines()
        with open(dataset_path + 'Test.csv') as f:
            test_file = f.readlines()

        self.train = train
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        datatype_path = 'faces/'
        
        for item in train_file:
            r = item.split(',')
            train_data.append(dataset_path + datatype_path + '/%s' % r[0])
            train_labels.append(r[1][:-1])

        for item in test_file:
            r = item.split(',')
            test_data.append(dataset_path + datatype_path + '/%s' % r[0])
            test_labels.append(r[1][:-1])

        dataset_path = '../dataset/imdb_crop/'

        with open(dataset_path + 'Train.csv') as f:
            train_file = f.readlines()
        with open(dataset_path + 'Test.csv') as f:
            test_file = f.readlines()

        datatype_path = 'faces/'
        
        for item in train_file:
            r = item.split(',')
            train_data.append(dataset_path + datatype_path + '/%s' % r[0])
            train_labels.append(r[1][:-1])

        for item in test_file:
            r = item.split(',')
            test_data.append(dataset_path + datatype_path + '/%s' % r[0])
            test_labels.append(r[1][:-1])


        self.data = None
        self.labels = None
        if self.train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels
            
    def __getitem__(self, index):
        img_path = self.data[index]
        img = load_image(img_path)
        if self.train:
            img = train_trans(img)
        else:
            img = test_trans(img)

        label = int(self.labels[index])
        class_label = label

        if self.train:
            return img, label, class_label
        else:
            return img, label, 1

    def __len__(self):
        return len(self.data)


class Logger():
    def __init__(self, title, save_path, append=False):
        print(save_path)
        print('================')
        file_state = 'wb'
        if append:
            file_state = 'ab'
        self.file = open(save_path, file_state, 0)
    
    def log(self, data):
        outstr = str(data) + '\n'
        outstr = outstr.encode("utf-8")
        self.file.write(outstr)

    def __del__(self):
        self.file.close()


