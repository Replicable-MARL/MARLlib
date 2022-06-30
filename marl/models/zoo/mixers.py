from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, states):
        return torch.sum(agent_qs, dim=2, keepdim=True)


class QMixer(nn.Module):
    def __init__(self, custom_config, state_dim):
        super(QMixer, self).__init__()

        self.n_agents = custom_config["num_agents"]
        self.raw_state_dim = state_dim
        self.embed_dim = custom_config["model_arch_args"]["mixer_embedding"]
        if len(state_dim) > 2:  # conv the state
            layers = []
            input_dim = state_dim[2]
            for i in range(custom_config["model_arch_args"]["conv_layer"]):
                conv_f = nn.Conv2d(
                    in_channels=input_dim,
                    out_channels=custom_config["model_arch_args"]["out_channel_layer_{}".format(i)],
                    kernel_size=custom_config["model_arch_args"]["kernel_size_layer_{}".format(i)],
                    stride=custom_config["model_arch_args"]["stride_layer_{}".format(i)],
                    padding=custom_config["model_arch_args"]["padding_layer_{}".format(i)],
                )
                relu_f = nn.ReLU()
                pool_f = nn.MaxPool2d(kernel_size=custom_config["model_arch_args"]["pool_size_layer_{}".format(i)])

                layers.append(conv_f)
                layers.append(relu_f)
                layers.append(pool_f)

                input_dim = custom_config["model_arch_args"]["out_channel_layer_{}".format(i)]
            self.extra_encoder = nn.Sequential(
                *layers
            )
            self.state_dim = input_dim
        else:
            self.extra_encoder = None
            self.state_dim = state_dim[0]

        if custom_config["global_state_flag"]:
            self.state_dim = self.state_dim
        else:
            self.state_dim = self.state_dim * self.n_agents

        self.hyper_w_1 = nn.Linear(self.state_dim,
                                   self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim), nn.ReLU(),
            nn.Linear(self.embed_dim, 1))

        self.custom_config = custom_config

    def forward(self, agent_qs, states):
        """Forward pass for the mixer.
        """
        bs = agent_qs.size(0)

        if self.extra_encoder:
            if self.custom_config["global_state_flag"]:
                x = states.reshape(-1, self.raw_state_dim[0], self.raw_state_dim[1],
                                   self.raw_state_dim[2]).permute(0, 3, 1, 2)
                x = self.extra_encoder(x)
                x = torch.mean(x, (2, 3))
                states = x.reshape(-1, self.state_dim)
            else:
                # for offpolicy the state size is 4
                if len(states.shape) == 4:
                    x = states.permute(2, 0, 1, 3).reshape(-1, self.raw_state_dim[0], self.raw_state_dim[1],
                                                           self.raw_state_dim[2]).permute(0, 3, 1, 2)
                elif len(states.shape) == 3:
                    # for onpolicy the state size is 3
                    x = states.permute(1, 0, 2).reshape(-1, self.raw_state_dim[0], self.raw_state_dim[1],
                                                        self.raw_state_dim[2]).permute(0, 3, 1, 2)

                else:
                    raise ValueError("wrong state shape")

                x = self.extra_encoder(x)
                x = torch.mean(x, (2, 3)).reshape(self.n_agents, -1, self.state_dim // self.n_agents)
                states = x.permute(1, 0, 2).reshape(-1, self.state_dim)
        else:
            states = states.reshape(-1, self.state_dim)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = nn.functional.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
