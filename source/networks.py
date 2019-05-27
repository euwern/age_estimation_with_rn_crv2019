import torch
import torch.nn as nn
import torchvision.models as models

class Model_Wiki(nn.Module):
    def __init__(self):
        super(Model_Wiki, self).__init__()

        base = models.__dict__['resnet50'](pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AvgPool2d(7, stride=1)
        
        self.fc1 = nn.Linear(2048, 1)
        self.fc2 = nn.Linear(2048, 101)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.base(input)
        output = self.pool(output).squeeze()
        output = self.relu(output)
        reg_out = self.fc1(output)
        class_out = self.fc2(output)
        reg_out = self.relu(reg_out)

        return reg_out, class_out

class Model_RN_Wiki(nn.Module):
    def __init__(self):
        super(Model_RN_Wiki, self).__init__()

        base = models.__dict__['resnet50'](pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AvgPool2d(49, stride=1)

        self.fc1 = nn.Linear(4096, 1)
        self.fc2 = nn.Linear(4096, 101)
        self.softmax = nn.Softmax(1) 
        self.relu = nn.ReLU()

    def forward(self, input):
        boutput = self.base(input)

        batch_size = input.size(0)
        boutput = boutput.permute(0,2,3,1).contiguous()
        boutput = boutput.view(batch_size, 49, 2048)
        boutput = boutput.unsqueeze(2).repeat(1, 1, 49, 1)
        
        boutput1 = boutput.view(batch_size, 49*49, 2048)
        boutput2 = boutput.permute(0,2,1,3).contiguous().view(batch_size, 49*49, 2048)

        boutput = torch.cat([boutput1, boutput2], 2)
        boutput = boutput.permute(0,2,1).contiguous()
        boutput = boutput.view(input.size(0), 4096, 49, 49)

        output2 = boutput
        output2 = self.pool(output2)
        output2 = output2.squeeze()
        output2 = self.relu(output2)
        reg_out = self.fc1(output2)
        class_out = self.fc2(output2)
        reg_out = self.relu(reg_out)

        return reg_out, class_out


class Model(nn.Module):
    def __init__(self, imdb_wiki_model_path=None):
        super(Model, self).__init__()

        if imdb_wiki_model_path == None:
            base = models.__dict__['resnet50'](pretrained=True)
            self.base = nn.Sequential(*list(base.children())[:-2])
            self.fc1 = nn.Linear(2048, 1)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(2048, 101)
          
        else:
            #training with pretrained model trained on WIKI or IMDB_WIKI
            temp_model = Model_Wiki()
            checkpoint = torch.load(imdb_wiki_model_path)
            temp_model.load_state_dict(checkpoint[0])
            self.base = temp_model.base
            self.fc1 = temp_model.fc1
            self.fc2 = temp_model.fc2
            self.relu = nn.ReLU()
        
        self.pool = nn.AvgPool2d(7, stride=1)


    def forward(self, input):
        output = self.base(input)
        output = self.pool(output).squeeze()
        output = self.relu(output)
        reg_output = self.fc1(output)
        class_output = self.fc2(output)
        reg_output = self.relu(reg_output)

        return reg_output, class_output

class Model_RN(nn.Module):
    def __init__(self, imdb_wiki_model_path=None):
        super(Model_RN, self).__init__()

        if imdb_wiki_model_path == None:
            base = models.__dict__['resnet50'](pretrained=True)
            self.base = nn.Sequential(*list(base.children())[:-2])
            self.fc1 = nn.Linear(4096, 1)
            self.fc2 = nn.Linear(4096, 101)
        else:
            #training with pretrained model trained on WIKI or IMDB_WIKI
            temp_model = Model_RN_Wiki()
            checkpoint = torch.load(imdb_wiki_model_path)
            temp_model.load_state_dict(checkpoint[0])
            self.base = temp_model.base
            self.fc1  = temp_model.fc1
            self.fc2  = temp_model.fc2

        self.pool = nn.AvgPool2d(49, stride=1)

        self.softmax = nn.Softmax(1) 
        self.relu = nn.ReLU()

    def forward(self, input):
        boutput = self.base(input)

        batch_size = input.size(0)
        boutput = boutput.permute(0,2,3,1).contiguous()
        boutput = boutput.view(batch_size, 49, 2048)
        boutput = boutput.unsqueeze(2).repeat(1, 1, 49, 1)
        
        boutput1 = boutput.view(batch_size, 49*49, 2048)
        boutput2 = boutput.permute(0,2,1,3).contiguous().view(batch_size, 49*49, 2048)

        boutput = torch.cat([boutput1, boutput2], 2)
        boutput = boutput.permute(0,2,1).contiguous()
        boutput = boutput.view(input.size(0), 4096, 49, 49)

        output2 = boutput
        output2 = self.pool(output2)
        output2 = output2.squeeze()
        output2 = self.relu(output2)
        reg_out = self.fc1(output2)
        class_out = self.fc2(output2)
        reg_out = self.relu(reg_out)

        return reg_out, class_out


