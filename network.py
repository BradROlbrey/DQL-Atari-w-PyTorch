
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=input_dims[0], out_channels=32, kernel_size=8, stride=4),
            # in_channels=4 corresponds to our stack of 4 frames.
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with T.no_grad():
            convs_output_dims = self.convs(T.zeros(1, *input_dims))
            linear_input_dims = int(np.prod(convs_output_dims.size()))

        self.linears = nn.Sequential(
            nn.Linear(linear_input_dims, 512),
            nn.ReLU(),

            nn.Linear(512, n_actions)
        )

        self.device = T.device('cuda')
        # self.device = T.device('cpu')  $ For better debug statements
        self.to(self.device)


    def forward(self, state):
        # Each conv layer is activated with a relu function, as per the paper.
        convs = self.convs(state)
        # batch_size (32), n_filters (64), H, W

        # Like I did in my pytorch_4 colab.
        # Go from  (BS x n_filters x H x W)  to  (BS x n_filters * H * W)
        flattened = convs.view(convs.size()[0], -1)

        # The output layer is just outputs.
        outputs = self.linears(flattened)

        return outputs
