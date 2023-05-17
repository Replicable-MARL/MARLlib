# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class QMixer(nn.Module):
    def __init__(self, custom_config, state_dim):
        super(QMixer, self).__init__()

        self.n_agents = custom_config["num_agents"]
        self.raw_state_dim = state_dim
        self.embed_dim = custom_config["model_arch_args"]["mixer_embedding"]
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
