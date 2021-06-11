import torch

class Attention(torch.nn.Module):
    """
        Calculates attention over the input nodes given the current state.
        Assuming that the GRU decoder's hidden state's dim is also num_neurons
        2*neurons because the hidden state is concatenated with the feature

        Assume that features are alread concatenation of static and dynamic_feature

    """


    def __init__(self, num_neurons, device):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.num_neurons = num_neurons
        self.v = torch.nn.Parameter(torch.randn((1, 1, num_neurons),
                                                device=device, requires_grad=True))

        self.W = torch.nn.Parameter(torch.randn((1, num_neurons, 3*num_neurons),
                                                device=device, requires_grad=True))

        self.to(device=device)

    def forward(self, features, pointer_hidden_state):

        batch_size, num_nodes, _ = features.shape
        # make shape similar for concatenation
        pointer_hidden_state = pointer_hidden_state.unsqueeze(1).expand(batch_size, num_nodes, self.num_neurons)
        hidden = torch.cat((features, pointer_hidden_state), 2)
        hidden = hidden.permute(0, 2, 1)
        # Broadcast some dimensions so we can do batch-matrix-multiply
        # i thought it means just changing the view
        # but it turns out, it is broadcasting.... (make several copies)
        v = self.v.expand(batch_size, -1, -1)
        W = self.W.expand(batch_size, -1, -1)
        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = torch.nn.functional.softmax(attns, dim=2)# (batch, seq_len)
        return attns
