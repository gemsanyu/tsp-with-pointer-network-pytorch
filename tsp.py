import torch

cpu_device = torch.device("cpu")

def get_tour_length_vectorized(tour_idx, W: torch.Tensor, device=cpu_device):
    """
        static def to compute tour length
        of many tours
        assuming that the tour is not started at 0 and ended at 0 yet
    """
    batch_size, num_nodes, _ = W.shape
    batch_idx = torch.arange(batch_size, dtype=torch.long, device=device)
    batch_idx = batch_idx.unsqueeze(1).expand(batch_size, num_nodes)
    tour_edge_list = torch.zeros(size=(batch_size, num_nodes, 3),
                                 dtype=torch.long,
                                 device=device)
    tour_edge_list[:, :, 0] = batch_idx
    tour_edge_list[:, :num_nodes-1, 2] = tour_idx
    tour_edge_list[:, 1:num_nodes, 1] = tour_idx
    distance_list = W[tour_edge_list[:,:,0],tour_edge_list[:,:,1], tour_edge_list[:,:,2]]
    return distance_list.sum(dim=1)


# so because the, i dont know what i was writing lol
def generate_graph(batch_size, num_nodes, low=0, high=20, device=cpu_device):
    coords = torch.randint(low=low, high=high, size=(batch_size ,num_nodes, 2),
                           device=device, dtype=torch.float32)
    W = torch.cdist(coords, coords, p=2).to(device)
    return coords, W


def get_greedy_tour(W, mode="min", device=cpu_device):
    """
        return tour_idx in tensor form, (batch_size, num_nodes-1)
        min => closest first
        max => furthest first
        random => random_tour
    """
    batch_size, num_nodes, _ = W.shape

    if mode == "random":
        tour_idx = torch.zeros(size=(batch_size, num_nodes-1), device=device,
                               dtype=torch.long)
        for i in range(batch_size):
            tour_idx[i,:]= torch.randperm(num_nodes-1, device=device) + 1
        return tour_idx, get_tour_length_vectorized(tour_idx, W, device=device)

    max_val = 99999999
    W_tmp = W.detach().clone()

    # set dist to node 0, and diagonal to maxval
    # to prevent other node to go to node 0 or itself
    diag_idx = torch.arange(num_nodes, device=device).repeat(batch_size)
    batch_idx = torch.arange(batch_size, device=device)
    batch_idx_ = batch_idx.repeat_interleave(num_nodes)
    W_tmp[batch_idx_, diag_idx, diag_idx] = max_val
    W_tmp[batch_idx_, diag_idx, 0] = max_val

    tour_idx = torch.zeros(size=(batch_size, num_nodes-1), dtype=torch.long,
                           device=device)
    current_node = torch.zeros(batch_size, dtype=torch.long)
    for i in range(num_nodes-1):
        next_node = W_tmp[batch_idx, current_node].argmin(dim=1)
        next_node_ = next_node.repeat_interleave(num_nodes)
        current_node = next_node
        W_tmp[batch_idx_, diag_idx, next_node_] = max_val
        tour_idx[batch_idx, i] = next_node

    return tour_idx, get_tour_length_vectorized(tour_idx, W)
