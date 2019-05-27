import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from tqdm import tqdm
import os
import argparse
import utils
import networks
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=int, default=1, help='split set')
parser.add_argument('--is_train', action='store_true', default=False)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--model', choices=['base', 'rn'], default='base')
parser.add_argument('--dataset', choices=['2015', '2016', 'wiki', 'imdb_wiki'], default='2015')
parser.add_argument('--imdb_wiki_model_path', type=str, default=None, help='path to model trained on WIKI or IMDB_WIKI dataset')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_steps', default=[10,15], nargs='+', type=int)


args = parser.parse_args()

save_path = 'results_' + args.model + '_' + args.dataset + '/'

if args.imdb_wiki_model_path != None:
    pretrained_dataset_name = os.path.dirname(arg.imdb_wiki_model_path).split('_')[-1]
    save_path += pretrained_dataset_name

if not os.path.exists(save_path):
    os.makedirs(save_path)

dataset_class = 'ImageDataset_' + args.dataset

def save_checkpoint():
    checkpoint = [model.state_dict(), opt.state_dict()]
    torch.save(checkpoint, '%s/checkpoint_%d_%s.pth' % (save_path, epoch, args.split))
    
def load_checkpoint(load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint[0])
    opt.load_state_dict(checkpoint[1])

def compute_abs_e(preds, targets):
    a = preds.squeeze() - targets.squeeze()
    return a.abs().sum()


def train():
    model.train()
    avg_loss = 0
    avg_acc = 0
    count = 0
    for _, (data, target, class_label) in enumerate(tqdm(data_loader)):
        opt.zero_grad()
        data, target, class_label  = Variable(data).cuda(), Variable(target.float()).cuda(), Variable(class_label).cuda()
        reg_out, class_out = model(data)
        prob = nn.Softmax(1)(class_out)
        exp = Variable(torch.Tensor(np.arange(0,101))).cuda()
        exp = exp.expand(data.size(0), 101)
        exp = prob * exp
        exp = exp.sum(1)
        
        loss = mse_loss(reg_out, target) + ent_loss(class_out, class_label) + mse_loss(exp, target)

        pred = exp
       
        loss.backward()
        opt.step()
        avg_loss = avg_loss + loss.data[0]
        avg_abs_e = avg_abs_e + compute_abs_e(pred.data, target.data)
        count += target.data.size(0)

    avg_loss = avg_loss / count
    avg_abs_e = avg_abs_e / count
    print('Epoch: %d; Loss: %f; MAE: %.2f' % (epoch, avg_loss, avg_abs_e))
    loss_logger.log(avg_loss)
    acc_logger.log(avg_acc)


def test():
    load_checkpoint('%s/checkpoint_%d_%d.pth' % (save_path, args.max_epochs, args.split))
    model.eval()
    data_loader = torch.utils.data.DataLoader(utils.__dict__[dataset_class](train=False), batch_size=args.batch_size)
    avg_acc = 0 
    count = 0
    pred_list = []
    gt_list = []
    gt2_list = []
    for _, (data, target, target2) in enumerate(tqdm(data_loader)):
        data, target  = Variable(data, volatile=True).cuda(), Variable(target.float(), volatile=True).cuda()
        reg_out,class_out = model(data)
        prob = nn.Softmax(class_out.dim()-1)(class_out)
        exp = Variable(torch.Tensor(np.arange(0,101))).cuda()
        exp = exp.expand(data.size(0), 101)
        exp = prob * exp
        exp = exp.sum(1)

        pred = exp
        curr_abs_e = compute_abs_e(pred.data, target.data)
        avg_abs_e = avg_abs_e + curr_abs_e

        pred_d = pred.data.squeeze()
        for ix in range(pred_d.size(0)):
            pred_list.append(pred_d[ix])
            gt_list.append(target.data[ix])
            gt2_list.append(target2[ix])

        count += target.data.size(0)
    preds = torch.Tensor(pred_list)
    gts = torch.Tensor(gt_list)
    gts2 = torch.Tensor(gt2_list)

    e_error = 1 - torch.exp(-1*(preds - gts)**2 / (2*gts2**2))
    print('MAE: %.3f; E-error: %.6f' % ((avg_abs_e / count), e_error.mean()))

if args.model == 'base' and (args.dataset == 'wiki' or args.datdaset == 'imdb_wiki'):
    model = networks.Model_Wiki()
elif args.model == 'rn' and (args.dataset == 'wiki' or args.dataset == 'imdb_wiki'):
    model = networks.Model_RN_Wiki()
elif args.model == 'base':    
    model = networks.Model(imdb_wiki_model_path=args.imdb_wiki_model_path)
elif args.model == 'rn':
    model = networks.Model_RN(imdb_wiki_model_path=args.imdb_wiki_model_path)

model = model.cuda()
data_loader = torch.utils.data.DataLoader(utils.__dict__[dataset_class](train=True), batch_size=args.batch_size, shuffle=True)
opt = optim.RMSprop([{'params': model.parameters(), 'lr': args.lr}])
sch = lr_scheduler.MultiStepLR(opt, milestones=args.lr_steps,  gamma=0.1)


if args.is_train:
    loss_logger = utils.Logger('loss', '{}/loss_{}.log'.format(save_path, args.split))
    acc_logger = utils.Logger('acc', '{}/acc_{}.log'.format(save_path, args.split))

mse_loss = nn.MSELoss().cuda()
ent_loss = nn.CrossEntropyLoss().cuda()

epoch = 1

if args.is_train:
    while True:
        train()
        if epoch % 1 == 0:
            save_checkpoint()
        if epoch == args.max_epochs:
            break
        sch.step(epoch)
        epoch = epoch + 1
    test()
else:
    test()
