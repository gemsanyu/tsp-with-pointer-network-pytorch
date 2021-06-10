import torch
from pointer import Pointer
from encoder import Encoder

class Actor(torch.nn.Module):
    def __init__(self, num_features, num_neurons, pointer_num_layers, learnable_first_input, device):
        super(Actor, self).__init__()

        self.num_features = num_features
        self.num_neurons = num_neurons
        self.device = device
        self.learnable_first_input = learnable_first_input

        self.static_encoder = Encoder(num_features, num_neurons, device)
        self.decoder_input_encoder = Encoder(num_features, num_neurons, device)
        self.pointer = Pointer(num_neurons, pointer_num_layers, device)
        self.first_input = torch.nn.Parameter(torch.randn(size=(1, 1, num_features), dtype=torch.float32,
                                                          device=device))
        self.to(device)

    """
        Forward means generating the routes, right away
        And the route will generate rewards + Expected Return
        So this is monte carlo learning
    """
    def forward(self, raw_features: torch.Tensor):
        batch_size, num_nodes, _ = raw_features.shape
        # row index 0->batch_size-1, for easier mask update
        row_idxs = torch.arange(batch_size)

        # First node always 0
        # the tour will not be started/ended by 0, because it's obvious
        # and to make the length similar to tour_logp
        if self.learnable_first_input:
            decoder_input = self.first_input.expand(batch_size, 1, self.num_features)
        else:
            decoder_input = raw_features[:, 0, :]
            decoder_input = decoder_input.unsqueeze(1)

        features = self.static_encoder(raw_features)
        tour_idx = torch.zeros(size=(batch_size, num_nodes-1),
                               device=self.device)
        tour_logp = torch.zeros(size=(batch_size, num_nodes-1),
                                device=self.device)
        mask = torch.ones(size=(batch_size, num_nodes),
                          dtype=torch.float32,
                          device=self.device)
        mask[:, 0] = 0

        last_pointer_hidden_state = None
        i = 0
        while torch.sum(mask) > 0:
            embedded_decoder_input = self.decoder_input_encoder(decoder_input)
            probs, last_pointer_hidden_state = self.pointer(features, embedded_decoder_input, last_pointer_hidden_state)

            # add log(mask) to probs, remember, log(0) in torch is defined as -Inf
            probs = torch.softmax(probs + mask.log(), dim=1)
            # if training, then sample from the probabilities
            # in testing, get the max
            if self.training:
                dist = torch.distributions.Categorical(probs)
                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                chosen_nodes = dist.sample()
                while not torch.gather(mask, 1, chosen_nodes.data.unsqueeze(1)).byte().all():
                    chosen_nodes = m.sample()
                logp = dist.log_prob(chosen_nodes)
            else:
                prob, chosen_nodes = torch.max(probs, 1)
                logp = prob.log()

            # Update the mask
            mask[row_idxs, chosen_nodes] = 0

            # get the chosen nodes' coords/features
            # as the input for the next RNN iteration
            gather_mask = chosen_nodes.view(batch_size, 1, 1).expand(batch_size, 1,  self.num_features)
            decoder_input = torch.gather(raw_features, 1, gather_mask)

            # add the chosen nodes to the tour
            # also the logprob for loss function
            tour_idx[:, i] = chosen_nodes.data
            tour_logp[:, i] = logp
            i = i + 1

        return tour_idx, tour_logp
