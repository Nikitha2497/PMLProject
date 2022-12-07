import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        # self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(4096, 512)
        # self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(512, 11)
        # self.bn3 = nn.BatchNorm1d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # TODO(nikitha) How to set linear input size??
        # linear_input_size = 16 * 32 * 32
        # self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        # x = x.to(device)
        print("inside forward ", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
