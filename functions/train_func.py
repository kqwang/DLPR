"""
Network training functions

From: https://github.com/kqwang/DLPR/
@author: Kaiqiang Wang
Email: kqwang.optics@gmail.com
"""
import torch
from functions.propagation import propagation

'Computes and stores the average and current value.'
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

' network training function for DD'
def train_net_DD(net, device, loader, optimizer, loss_f):
    net.train()
    train_loss = AverageMeter()
    for batch_idx, (input, gt) in enumerate(loader):
        input, gt = input.to(device), gt.to(device)  # Send data to GPU
        output = net(input)  # Network forward
        loss = loss_f(output, gt)  # Loss calculation
        train_loss.update(loss.item(), output.size(0))  # Update the record

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss.avg, input, output, gt

' network training function for PD'
def train_net_PD(net, device, loader, optimizer, loss_f, prop_dis=20, norm=False, rand_to_holo=False, dim=256, pad=False):
    net.train()
    train_loss = AverageMeter()
    for batch_idx, (input, gt) in enumerate(loader):
        input, gt = input.to(device), gt.to(device) # Send data to GPU

        # Add rand to hologram
        if rand_to_holo:
            rand = torch.rand((1, 1, dim, dim)) * (1.0/30)
            input = input + rand.to(device)

        output = net(input)  # network forward
        H = propagation(output, prop_dis=prop_dis, norm=norm, dim=dim, pad=pad)
        loss = loss_f(input, H)  # Loss calculation
        train_loss.update(loss.item(), output.size(0))  # Update the record

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss.avg, input,  output, gt

' network training function for CD'
def train_net_CD(net, device, loader, optimizer, loss_f, prop_dis=20, norm=False, rand_to_holo=False, dim=256, alpha=0.3, pad=False):
    net.train()
    train_loss = AverageMeter()
    for batch_idx, (input, gt) in enumerate(loader):
        input, gt = input.to(device), gt.to(device) # Send data to GPU

        # Add rand to hologram
        if rand_to_holo:
            rand = torch.rand((1, 1, dim, dim)) * (1.0/30)
            input = input + rand.to(device)

        output = net(input)  # network forward
        H = propagation(output, prop_dis=prop_dis, norm=norm, dim=dim, pad=pad)
        loss0 = loss_f(output, gt)  # Loss calculation of DD
        loss1 = loss_f(input, H)  # Loss calculation of PD
        loss = alpha*loss0 + loss1
        train_loss.update(loss.item(), output.size(0))  # Update the record

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss.avg, input,  output, gt