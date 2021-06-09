import pathlib

import torch
from actor import Actor
from critic import Critic

cpu_device = torch.device("cpu")


class Agent(torch.nn.Module):
    def __init__(self,
                 num_features=2,
                 num_neurons=32,
                 critic_num_layers=2,
                 pointer_num_layers=2,
                 critic_num_neurons=20,
                 learning_rate=3e-4,
                 max_grad=10,
                 eps=None,
                 min_eps=None,
                 eps_decay=None,
                 learnable_first_input=False,
                 device=cpu_device):

        super(Agent, self).__init__()

        self.device = device
        self.num_features = 2
        self.max_grad = max_grad
        self.actor = Actor(num_features=2, num_neurons=num_neurons, learnable_first_input=learnable_first_input,
                           pointer_num_layers=2, eps=eps, min_eps=min_eps, eps_decay=eps_decay,
                           device=device)
        self.critic = Critic(num_features=2, encoder_num_neurons=num_neurons, critic_num_neurons=critic_num_neurons,
                             device=device, num_layers=critic_num_layers)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.to(self.device)

    def forward(self, raw_features, distance_matrix, is_training=True):
        if not is_training:
            with torch.no_grad():
                tour_idx, tour_logp = self.actor(raw_features)
        else:
            tour_idx, tour_logp = self.actor(raw_features)

        return tour_idx, tour_logp

    def optimize(self, raw_features, tour_logp, tour_length):
        """
            Calculate advantage = (tour_length - critic_length)
            if tour_length > critic_length, then advantage > 0
            , the bigger the tour_length, the bigger advantage,
            the bigger the advantage, the bigger the loss,
            remember This is intended, because
            the optimizer (ADAM, SGD, etc) is supposed to minimize loss
            so, the optimizer will minimize the difference between
            tour_length and critic_length,
            and because of the critic's weights are initiated
            near zero, then it will be minimized
        """
        critic_length = self.critic(raw_features)
        critic_length = critic_length.squeeze(1)
        advantage = (tour_length - critic_length)

        actor_loss = torch.mean(advantage.detach()*tour_logp.sum(dim=1))
        critic_loss = torch.mean(advantage**2)

        # Clear previous grads, and optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
        self.critic_optimizer.step()

        return actor_loss.data, critic_loss.data

    def save(self, save_dir, prefix=None, suffix=None):
        actor_file_name = "actor"
        if prefix is not None:
            actor_file_name = prefix + "_" + actor_file_name
        if suffix is not None:
            actor_file_name = actor_file_name + "_" + suffix
        actor_file_name = actor_file_name +".pt"

        critic_file_name = "critic"
        if prefix is not None:
            critic_file_name = prefix + "_" + critic_file_name
        if suffix is not None:
            critic_file_name = critic_file_name + "_" + suffix
        critic_file_name = critic_file_name +".pt"

        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        actor_save_path = save_dir/actor_file_name
        critic_save_path = save_dir/critic_file_name

        torch.save(self.actor.state_dict(), actor_save_path)
        print(actor_file_name, "saved successfully")
        torch.save(self.critic.state_dict(), critic_save_path)
        print(critic_file_name, "saved successfully")

    def load_actor(self, load_dir, file_name):
        load_dir = pathlib.Path(load_dir)
        if not pathlib.Exists(load_dir):
            print("ERROR: directory not exists:", load_dir.name)
            return
        load_path = load_dir/file_name
        self.actor.load_state_dict(load_path, map_location=self.device)

    def load_critic(self, load_dir, file_name):
        load_dir = pathlib.Path(load_dir)
        if not pathlib.Exists(load_dir):
            print("ERROR: directory not exists:", load_dir.name)
            return
        load_path = load_dir/file_name
        self.critic.load_state_dict(load_path, map_location=self.device)
