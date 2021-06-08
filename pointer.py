import torch

from attention import Attention

"""
    Pointer to output probability
    and also the last hidden feature from the RNN (GRU)
    Assuming the input to the GRU is the HIDDEN FEATURE (ENCODED) state of the previously selected node
    in this case ENCODE(coords)
"""

class Pointer(torch.nn.Module):
    def __init__(self, num_neurons, num_layers, device, dropout=0.2):
        super(Pointer, self).__init__()

        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.v = torch.nn.Parameter(torch.randn((1, 1, num_neurons),
                                                device=device, requires_grad=True))

        self.W = torch.nn.Parameter(torch.randn((1, num_neurons, 2*num_neurons),
                                                device=device, requires_grad=True))

        if num_layers == 1:
            dropout = 0
        self.gru = torch.nn.GRU(num_neurons, num_neurons, num_layers, batch_first=True, dropout=dropout)

        self.attention_layer = Attention(num_neurons, device)
        self.drop_rnn = torch.nn.Dropout(p=dropout)
        self.drop_hh = torch.nn.Dropout(p=dropout)

    """
        Features = Encoded Features of all node (batch_size, N, num_neurons)
        Decoder_input = input of GRU, currently is the encoded feature of the previously selected node (batch_size, num_neurons)
        Last_pointer_hidden_state = literally last hidden state from the previous node selection (batch_size, num_neurons)
    """
    def forward(self, features, decoder_input, last_pointer_hidden_state):
        batch_size, num_nodes, _ = features.shape

        rnn_out, pointer_hidden_state = self.gru(decoder_input, last_pointer_hidden_state)
        rnn_out = rnn_out.squeeze(1) # (batch_size, 1, num_neurons)-> (batch_size, num_neurons)
        rnn_out = self.drop_rnn(rnn_out)

        # now, we will use the output of the RNN to compute the attention
        # the attentions now will be used to get the context feature (weighted sum of features)
        attentions = self.attention_layer(features, rnn_out)
        contexts = torch.bmm(attentions, features)

        # Now the context is used to get energy
        # energy = Vp * tanh(Wp[Features;context])
        # energy is like attention (weights summed to 1),
        # and gonna be directly used to determin the probability
        # broadcast V and W, so we can use batch mat-mul (bmm)
        contexts = contexts.expand_as(features)
        feature_contexts = torch.cat((features, contexts), dim=2)
        feature_contexts = feature_contexts.permute(0,2,1)
        v = self.v.expand(batch_size, -1, -1)
        W = self.W.expand(batch_size, -1, -1)
        energy = torch.bmm(v, torch.tanh(torch.bmm(W, feature_contexts)))
        energy = energy.squeeze(1)

        return energy, pointer_hidden_state
