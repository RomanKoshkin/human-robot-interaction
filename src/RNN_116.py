import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, output_size):
        super(RNN, self).__init__()                  # extend the functionality of previously built classes.
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        # YOU CAN USE EITHER LSTM, GRU OR VANILLA RNN
        if self.cell_type=='GRU':
            self.rnn_cell = nn.GRU(input_size=input_size,    # the dimensionality of ONE ELEMENT in a sequence
                                hidden_size=hidden_size, # apparently the hidden state and output dimensionality must be the same
                                num_layers=1)            # how many LSTM cells we want to stack (defalult=1)
        if self.cell_type=='LSTM':
            self.rnn_cell = nn.LSTM(input_size=input_size,    # the dimensionality of ONE ELEMENT in a sequence
                                hidden_size=hidden_size, # apparently the hidden state and output dimensionality must be the same
                                num_layers=1)            # how many LSTM cells we want to stack (defalult=1)
        if self.cell_type=='RNN':
            self.rnn_cell = nn.RNN(input_size=input_size,    # the dimensionality of ONE ELEMENT in a sequence
                                hidden_size=hidden_size, # apparently the hidden state and output dimensionality must be the same
                                num_layers=1)            # how many LSTM cells we want to stack (defalult=1)

        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, padding=(1,1))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(7, 28, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 =   nn.Linear(7168, 100)
        self.fc2 =   nn.Linear(100, 30)
        self.fc3 =   nn.Linear(46, 16) # latent video (10) + motor (2) output from RNN to motor prediction
        self.fc4 =   nn.Linear(46, 64)
        self.us1 =   nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.us2 =   nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.us3 =   nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.us4 =   nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.us5 =   nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.conv3 = nn.Conv2d(1, 10, 3, padding=1)
        self.conv4 = nn.Conv2d(10, 20, 3, padding=1)
        self.conv5 = nn.Conv2d(20, 1, 3, padding=1)
#         self.optimizer = optim.RMSprop(self.parameters(),
#                                        lr=0.001,
#                                        momentum=0.00,
#                                        weight_decay=0.000,
#                                        centered=False)
        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)        
        
    def forward(self, visual_input, motor_input, hidden):
        
        # visual pathway:
#         print('\n\nEncoding video into a latent representation')
        out = F.relu(self.conv1(visual_input))
#         print('after conv 1: ', out.shape)
        out = self.pool1(out)
#         print('after pool 1: ', out.shape)
        out = F.relu(self.conv2(out)) # torch.Size([1367, 15, 58, 58])
#         print('after conv 2: ', out.shape)
        out = self.pool2(out)         # torch.Size([1367, 15, 28, 28])
#         print('after pool 2: ', out.shape)
        out = out.view(-1, out.size()[1]*out.size()[2]*out.size()[3]) # flatten
#         print('after flatten: ', out.shape)
        out = F.relu(self.fc1(out))
#         print('after fc 1: ', out.shape)
        out = F.relu(self.fc2(out))
#         print('after fc 2: ', out.shape)
        
        # concatenate visual pathway and motor input:
        out = torch.cat((out, motor_input), dim=1).unsqueeze_(1) # unsqueeze adds a dimension (for batch=1) inplace
#         print('after concatenating vis and motor: ', out.shape)
        
        # run this combined input through an RNN cell (to predict the next visual input and motor state):
        out, hidden = self.rnn_cell(out, hidden)
        
        # predict motor output based on the latent representation:
        motor_output = torch.tanh(self.fc3(out.squeeze()))
#         print('after fc 3 (motor output based on rnn cell): ', motor_output.shape)
        
        # reconstruct video from the latent representation:
#         print('\n\nReconstruction of video')
        out1 = F.relu(self.fc4(out.squeeze()))
#         print('after fc 4: ', out1.shape)
        out1 = out1.view(-1,1,8,8)
#         print('after reshaping: ', out1.shape)
        out1 = self.us1(out1)
#         print('after us 1: ', out1.shape)
        out1 = F.relu(self.conv3(out1))
#         print('after conv 3: ', out1.shape)
        out1 = self.us2(out1)
#         print('after us 2: ', out1.shape)
        out1 = F.relu(self.conv4(out1))
#         print('after conv 4: ', out1.shape)
        out1 = self.us3(out1)
#         print('after us 3: ', out1.shape)
        out1 = F.relu(self.conv5(out1))
#         print('after conv 5: ', out1.shape)

        return out1, motor_output, hidden

    def initHidden(self, device):
            if self.cell_type=='LSTM': # we initialize a 2-tuple of hidden states (hidden state, memory)
                return (torch.zeros(1, 1, self.hidden_size).to(device), torch.zeros(1, 1, self.hidden_size).to(device))
            if self.cell_type=='GRU' or self.cell_type=='RNN': # we initialize a hidden state
                return torch.zeros(1, 1, self.hidden_size).to(device)

# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------

class vaeRNN(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, output_size):
        super(vaeRNN, self).__init__()                  # extend the functionality of previously built classes.
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        # YOU CAN USE EITHER LSTM, GRU OR VANILLA RNN
        if self.cell_type=='GRU':
            self.rnn_cell = nn.GRU(input_size=input_size,    # the dimensionality of ONE ELEMENT in a sequence
                                hidden_size=hidden_size, # apparently the hidden state and output dimensionality must be the same
                                num_layers=1)            # how many LSTM cells we want to stack (defalult=1)
        if self.cell_type=='LSTM':
            self.rnn_cell = nn.LSTM(input_size=input_size,    # the dimensionality of ONE ELEMENT in a sequence
                                hidden_size=hidden_size, # apparently the hidden state and output dimensionality must be the same
                                num_layers=1)            # how many LSTM cells we want to stack (defalult=1)
        if self.cell_type=='RNN':
            self.rnn_cell = nn.RNN(input_size=input_size,    # the dimensionality of ONE ELEMENT in a sequence
                                hidden_size=hidden_size, # apparently the hidden state and output dimensionality must be the same
                                num_layers=1)            # how many LSTM cells we want to stack (defalult=1)

        self.conv1a = nn.Conv2d(1, 7, kernel_size=3, padding=(1,1))
        self.conv1b = nn.Conv2d(7, 7, kernel_size=3, padding=(1,1))
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2a = nn.Conv2d(7, 28, 3, padding=1)
        self.conv2b = nn.Conv2d(28, 28, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 =   nn.Linear(7168, 500)
        self.fc2 =   nn.Linear(500, 100)
        self.fc3 =   nn.Linear(116, 16) # latent video (10) + motor (2) output from RNN to motor prediction
        self.fc4 =   nn.Linear(116, 256)
        self.us1 =   nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.us2 =   nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.us3 =   nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.us4 =   nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.us5 =   nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.fcmu = nn.Linear(116, 116)
        self.fcsig = nn.Linear(116, 116)

        self.conv3a = nn.Conv2d(1, 10, 3, padding=1)
        self.conv3b = nn.Conv2d(10, 10, 3, padding=1)
        self.conv4a = nn.Conv2d(10, 20, 3, padding=1)
        self.conv4b = nn.Conv2d(20, 20, 3, padding=1)
        self.conv5a = nn.Conv2d(20, 1, 3, padding=1)
        self.conv5b = nn.Conv2d(1, 1, 3, padding=1)
#         self.optimizer = optim.RMSprop(self.parameters(),
#                                        lr=0.001,
#                                        momentum=0.00,
#                                        weight_decay=0.000,
#                                        centered=False)
        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)        
    
    def reparameterize(self, mu, logsig, train):
        std = torch.exp(logsig)
        eps = torch.randn_like(std)
        return mu + eps*std if train else mu   

    def KL(self, mu, logsig):
        Dkl = - 0.5 * torch.sum(1 + logsig - mu.pow(2) - logsig.exp())
        return Dkl 
    
    def forward(self, visual_input, motor_input, hidden, train):
        # visual pathway:
        out = F.relu(self.conv1a(visual_input))                          # torch.Size([?, 7, 64, 64])
        out = F.relu(self.conv1b(out))                          # torch.Size([?, 7, 64, 64])
        out = self.pool1(out)                                           # torch.Size([?, 7, 32, 32])
        out = F.relu(self.conv2a(out))                                   # torch.Size([?, 28, 32, 32])
        out = F.relu(self.conv2b(out))                                   # torch.Size([?, 28, 32, 32])
        out = self.pool2(out)                                           # torch.Size([?, 28, 16, 16])
        out = out.view(-1, out.size()[1]*out.size()[2]*out.size()[3])   # torch.Size([?, 7168])
        out = F.relu(self.fc1(out))                                     # torch.Size([?, 500])
        out = F.relu(self.fc2(out))                                     # torch.Size([?, 100])
        
        # concatenate visual pathway and motor input. unsqueeze adds a dimension (for batch=1) inplace:
        out = torch.cat((out, motor_input), dim=1)                      # torch.Size([?, 46])
              
        mu = self.fcmu(out)                                   # torch.Size([?, 46])      
        logsig = self.fcsig(out)                              # torch.Size([?, 46])
        
        latent = self.reparameterize(mu, logsig, train=train)  # torch.Size([?, 1, 46])

        # run this combined input through an RNN cell (to predict the next visual input and motor state):
        latent, hidden = self.rnn_cell(latent.unsqueeze(1), hidden)                        # torch.Size([?, 1, 46])

        # predict motor output based on the latent representation:
        motor_output = torch.tanh(self.fc3(latent.squeeze(1)))               # torch.Size([?, 16])
        
        # reconstruct video from the latent representation:
        out1 = F.relu(self.fc4(latent.squeeze(1)))                          # torch.Size([?, 64])
        out1 = out1.view(-1,1,16,16)                                      # torch.Size([?, 1, 8, 8])
        out1 = self.us1(out1)                                           # torch.Size([?, 1, 16, 16])
        out1 = F.relu(self.conv3a(out1))                                 # torch.Size([?, 10, 16, 16])
        out1 = F.relu(self.conv3b(out1))                                 # torch.Size([?, 10, 16, 16])
        out1 = self.us2(out1)                                           # torch.Size([?, 10, 32, 32])
        out1 = F.relu(self.conv4a(out1))                                 # torch.Size([?, 20, 32, 32])
        out1 = F.relu(self.conv4b(out1))                                 # torch.Size([?, 20, 32, 32])
        # out1 = self.us3(out1)                                           # torch.Size([?, 20, 64, 64])
        out1 = torch.relu(self.conv5a(out1))                             # torch.Size([?, 1, 64, 64])
        out1 = torch.relu(self.conv5b(out1))                             # torch.Size([?, 1, 64, 64])

        return out1, motor_output, hidden, mu, logsig

    def initHidden(self, device):
            if self.cell_type=='LSTM': # we initialize a 2-tuple of hidden states (hidden state, memory)
                return (torch.zeros(1, 1, self.hidden_size).to(device), torch.zeros(1, 1, self.hidden_size).to(device))
            if self.cell_type=='GRU' or self.cell_type=='RNN': # we initialize a hidden state
                return torch.zeros(1, 1, self.hidden_size).to(device)