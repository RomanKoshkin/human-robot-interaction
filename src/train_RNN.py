import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# from RNN_116 import vaeRNN, RNN       # 116 units
from RNN import vaeRNN, RNN             # 46 units


import sys
import time
import random
import numpy as np
import cv2  # conda install opencv
import pickle


import matplotlib.pyplot as plt
from tabulate import tabulate
from utils import printProgressBar


def loadset(device):
    fname = np.random.choice(['DS0.dat', 'DS1.dat','DS2.dat','DS3.dat'])

    with open(fname, 'rb') as f:
        dat = pickle.load(f)

    IMG = dat['img']
    joints = dat['joints']
    times = dat['t']

    joints /= 4.0

    img = np.stack(IMG)
    img = img.transpose(0,3,1,2) #B x C x H x W
    img = img[:,0,:,:]           # select only the red channel
    img = img[:,None, :,:]       # squeeze the tensor
    visual_input = torch.tensor(img, dtype=torch.float).to(device)

    # prepare visual targets (they should be the same as the inputs only one step ahead)
    visual_target = np.roll(img, -1, axis=0)
    visual_target = torch.tensor(visual_target, dtype=torch.float).to(device)

    motor_input = torch.tensor(joints, dtype=torch.float).to(device)

    target_joints = np.roll(joints, -1, axis=0)
    motor_target = torch.tensor(target_joints, dtype=torch.float).to(device)
    return IMG, joints, visual_input, motor_input, visual_target, motor_target

input_size  = 116
hidden_size = 116
output_size = 116
cell_type = 'GRU'

EPOCHS = 50000
device = 'cuda'

criterion = nn.MSELoss()

rnn = vaeRNN(cell_type, input_size, hidden_size, output_size).to(device)

# rnn.load_state_dict(torch.load('checkpoint'))

hidden = rnn.initHidden(device=device)
IMG, joints, visual_input, motor_input, visual_target, motor_target = loadset(device)
visual_output, motor_output, hidden, mu, logsig = rnn(visual_input, motor_input, hidden, True)

VLOSS, MLOSS = [], []
name = ''.join(random.sample('abcdefgh1234567890', 10))
writer = SummaryWriter('runs/' + name)


""" with prediction feedback """


rnn.optimizer = optim.Adam(rnn.parameters(), lr=0.0005)
# rnn.optimizer = optim.Adam(rnn.parameters())
pred_fb = 0.1                                 # fraction of feedback at each step (ground truth = 1 - pred_fb)

printProgressBar(0, EPOCHS, prefix = 'Progress:', suffix = 'Complete', length = 25)
for epoch in range(EPOCHS):
    try:
        if epoch%5 == 0:
            IMG, joints, visual_input, motor_input, visual_target, motor_target = loadset(device)
        
        # forward the first step in the sequence:
        hidden = rnn.initHidden(device=device)
        visual_output, motor_output, hidden, mu, logsig =   rnn(visual_input[0].view(-1,1,64,64),
                                                            motor_input[0].view(-1,16),
                                                            hidden)  
        rnn.optimizer.zero_grad()
        loss = torch.zeros(1,).to(device)
        
        for i in range(1, motor_input.shape[0]):
            visual_output, motor_output, hidden, mu, logsig =   rnn(visual_input[i].view(-1,1,64,64)*(1-pred_fb) + visual_output*pred_fb,
                                                                motor_input[i].view(-1,16)*(1-pred_fb) + motor_output.view(-1,16)*pred_fb,
                                                                hidden)

            motor_loss = criterion(motor_output.view(-1,16), motor_target[i].view(-1,16))
            visual_loss = criterion(visual_output, visual_target[i].view(-1,1,64,64))
            kl_loss = rnn.KL(mu, logsig)
            
            total_loss_ = visual_loss.item() + motor_loss.item() + kl_loss.item()
            visual_mult = total_loss_ / 3 / visual_loss.item()
            motor_mult = total_loss_ / 3 / motor_loss.item()
            Dkl_mult = total_loss_ / 3 / kl_loss.item()
            loss += visual_loss*visual_mult + motor_loss*motor_mult + kl_loss*Dkl_mult

        loss.backward()
        writer.add_scalar('motor loss', motor_loss, epoch)
        writer.add_scalar('visual loss', visual_loss, epoch)
            
        VLOSS.append(visual_loss.item())
        MLOSS.append(motor_loss.item())
        rnn.optimizer.step()
        printProgressBar(epoch + 1, EPOCHS, prefix='Epoch: {} vloss: {:.2f} mloss: {:.6f} Dkl: {:.3f}'.format(epoch, visual_loss.item(), motor_loss.item(), kl_loss.item()), suffix='Complete', length=25)
        if epoch % 1000 == 0:
            torch.save(rnn.state_dict(), 'checkpoint')

    except KeyboardInterrupt:
        print('\nKeyboard Interrupt')
        break
        