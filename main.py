import torch
from torch.utils.tensorboard import SummaryWriter

import pathlib

from agent import Agent
from tsp import get_tour_length_vectorized
from tsp import generate_graph
from tsp import get_greedy_tour

device = torch.device("cpu")
max_epoch = 50000
batch_size= 128
min_graph_size = 10
max_graph_size = 40

num_neurons = 32
critic_num_neurons = 20
critic_num_layers = 2
pointer_num_layers= 2

learning_rate = 3e-4
max_grad = 10

title="run_small_randn"
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
              device=device)

# visualizing the net
# coords, W = generate_graph(batch_size, 5, device=device)
# writer.add_graph(agent.actor, coords)
# writer.add_graph(agent.critic, coords)
# writer.close()

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


        # maybe track the weights of each parameter too ??
        # CRITIC
        # for name, param in agent.critic.named_parameters():
        #     if param.requires_grad:
        #         print("CRITIC/"+name, param)
        #         print("GRAD",param.grad)

        # for name, param in agent.actor.named_parameters():
        #     if param.requires_grad:
        #         print("ACTOR/"+name, param)
        #         print("GRAD",param.grad)
    # write epoch average improvement
    writer.add_scalar("Epoch_Improvement", total_improvement/n_iter, epoch+1)

    if total_improvement/n_iter > best_average_improvement:
        best_average_improvement = total_improvement/n_iter
        agent.save(checkpoint_dir,suffix=str(epoch+1))
