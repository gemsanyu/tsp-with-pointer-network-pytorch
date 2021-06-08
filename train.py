import torch
from torch.utils.tensorboard import SummaryWriter

import pathlib

from agent import Agent
from tsp import get_tour_length_vectorized
from tsp import generate_graph
from tsp import get_greedy_tour

DEVICE = torch.device("cpu")
# DEVICE = torch.device("gpu" if torch.is_available_gpu() else "cpu")

@click.command()
@click.option('--max-epoch', default=10000, help="Number of epoch", type=int)
@click.option('--max-epoch', default=128, help="Size of batch per iteration", type=int)
@click.option('--min-graph-size', default=10, help="THe smallest size of training graph", type=int)
@click.option('--max-graph-size', default=50, help="THe smallest size of training graph", type=int)
@click.option('--num-neurons', default=32, help="Number of neurons for every layer, except critic", type=int)
@click.option('--critic-num-neurons', default=20, help="number of neurons in critic layer", type=int)
@click.option('--critic-num-layers', default=2, help="Number of layers in critic MLP", type=int)
@click.option('--pointer-num-layers', default=2, help="Number of layers in Pointer's GRU", type=int)
@click.option('--learning-rate', default=3e-4, help="optimizer learning rate", type=float)
@click.option('--max-grad', default=10, help="max gradient for gradient clipping", type=float)
@click.option('--learnable-first-input', default=False, help="wether use parameters or node 0 as first input", type=bool)
@click.option('--title', default="not_learnable", help="title for saving and tracking", type=str)
def train(max_epoch, batch_size, min_graph_size, max_graph_size, num_neurons, critic_num_neurons,
          critic_num_layers, pointer_num_layers, learning_rate, max_grad, learnable_first_input, title):

    device = DEVICE

    # Prepare checkpoint directory and tensorboard directory
    checkpoint_root="checkpoint"
    checkpoint_dir=pathlib.Path(".")/checkpoint_root/title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary_root="runs"
    summary_dir=pathlib.Path(".")/summary_root/title
    summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=summary_dir.absolute())

    agent = Agent(num_features=2,
                  num_neurons=num_neurons,
                  critic_num_layers=critic_num_layers,
                  pointer_num_layers=pointer_num_layers,
                  critic_num_neurons=critic_num_neurons,
                  learning_rate=learning_rate,
                  max_grad=max_grad,
                  learnable_first_input=learnable_first_input,
                  device=device)

    # # 1 epoch = n iter
    # # n = max_num_nodes - min_num_nodes
    # # every epoch check if average improvement is better than last epoch
    # # then save the model
    best_average_improvement = -999
    n_iter = max_graph_size-min_graph_size+1
    i = 0
    for epoch in range(max_epoch):
        total_improvement = 0.
        for num_nodes in range(min_graph_size, max_graph_size+1):
            coords, W = generate_graph(batch_size, num_nodes, device=device)
            tour_idx, tour_logp = agent.forward(raw_features=coords,
                                                     distance_matrix=W,
                                                     is_training=True)
            tour_length = get_tour_length_vectorized(tour_idx, W)
            actor_loss, critic_loss = agent.optimize(coords, tour_logp, tour_length)
            greedy_tour_idx, greedy_tour_length = get_greedy_tour(W, mode="min",
                                                                  device=device)
            random_tour_idx, random_tour_length = get_greedy_tour(W, mode="random",
                                                                  device=device)
            improvement = ((greedy_tour_length - tour_length)/greedy_tour_length).mean()*100
            improvement_random = ((random_tour_length - tour_length)/random_tour_length).mean()*100
            total_improvement += improvement.item()
            i += 1

            # write to summary, improvement, losses
            writer.add_scalar("Improvement_from_greedy", improvement, i)
            writer.add_scalar("Improvement_from_random", improvement_random, i)
            writer.add_scalar("Actor_Loss", actor_loss, i)
            writer.add_scalar("Critic_Loss", critic_loss, i)

        # write epoch average improvement
        writer.add_scalar("Epoch_Improvement", total_improvement/n_iter, epoch+1)

        if total_improvement/n_iter > best_average_improvement:
            best_average_improvement = total_improvement/n_iter
            agent.save(checkpoint_dir,suffix=str(epoch+1))


if __name__=='__main__':
    train()