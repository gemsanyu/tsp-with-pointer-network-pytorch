import torch

class Encoder(torch.nn.Module):
    def __init__(self, input_size, num_neurons, device):
        super(Encoder, self).__init__()
        self.device = device
        self.layer = torch.nn.Linear(input_size, num_neurons)
        self.to(device=self.device)

    def forward(self, x):
        output = self.layer(x)
        return output
