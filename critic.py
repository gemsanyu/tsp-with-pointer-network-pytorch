import torch
from encoder import Encoder

# Critic will give estimated length, given the coords
# Critic has its own encoder for the raw features
class Critic(torch.nn.Module):
    def __init__(self, num_features, encoder_num_neurons, critic_num_neurons, device, num_layers=2):
        super(Critic, self).__init__()

        self.num_features = num_features
        self.num_layers = num_layers

        self.encoder = Encoder(num_features, encoder_num_neurons, device)
        self.layer_list = [torch.nn.Linear(encoder_num_neurons, critic_num_neurons)]
        for i in range(num_layers-2):
            self.layer_list += [torch.nn.Linear(critic_num_neurons, critic_num_neurons)]
        self.layer_list += [torch.nn.Linear(critic_num_neurons, 1)]
        self.layer_list = torch.nn.ModuleList(self.layer_list)
        for p in self.parameters():
            if len(p.shape)>1:
                torch.nn.init.xavier_uniform_(p)
        self.to(device)

    # last layer without activation, and then summed of all nodes
    # remember, the input is in size (batch_size, num_nodes, raw_features)
    # in last layer, it becomes, (batch_size, num_nodes, 1)
    # while we want (batch_size, 1), thus the sum
    def forward(self, raw_features):
        features = self.encoder(raw_features)
        output = features
        for i in range(self.num_layers-1):
            output = torch.nn.functional.relu(self.layer_list[i](output))
        output = self.layer_list[-1](output).sum(1)
        return output
