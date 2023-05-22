import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size) -> None:
        super().__init__()
        self._size = [input_dim] + list(map(int, hidden_size.split(' ')))
        self._hidden_layers = len(self._size) - 1
        active_func = nn.ReLU()

        fc_h = []
        for j in range(len(self._size) - 1):
            fc_h += [
                nn.Linear(self._size[j], self._size[j + 1]), active_func, nn.LayerNorm(self._size[j + 1])
            ]
        self.fc = nn.Sequential(*fc_h)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPBase(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.mlp = MLPLayer(input_dim, hidden_size)

    def forward(self, x):
        x = self.mlp(x)
        return x


class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers)
        # NOTE: self.gru(x, hxs) needs x=[T, N, input_size] and hxs=[L, N, hidden_size]
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: F.Tensor, hxs: F.Tensor):
        # x=[N, input_size], hxs=[N, L, hidden_size]
        x, hxs = self.gru(x.unsqueeze(0), hxs.transpose(0, 1).contiguous())
        x = x.squeeze(0)            # [1, N, input_size] => [N, input_size]
        hxs = hxs.transpose(0, 1)   # [L, N, hidden_size] => [N, L, hidden_size]
        x = self.norm(x)
        return x, hxs


class Categorical(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Categorical, self).__init__()
        self.logits_net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.logits_net(x)
        return torch.distributions.Categorical(logits=logits).probs.argmax(dim=-1, keepdim=True)


class ACTLayer(nn.Module):
    def __init__(self, input_dim, action_dims, use_mlp_actlayer=False):
        super(ACTLayer, self).__init__()
        self._mlp_actlayer = use_mlp_actlayer
        if self._mlp_actlayer:
            self.mlp = MLPLayer(128, '128 128')
        action_outs = []
        for action_dim in action_dims:
            action_outs.append(Categorical(input_dim, action_dim))
        self.action_outs = nn.ModuleList(action_outs)

    def forward(self, x):
        if self._mlp_actlayer:
            x = self.mlp(x)
        actions = []
        for action_out in self.action_outs:
            action = action_out(x)
            actions.append(action)
        actions = torch.cat(actions, dim=-1)
        return actions


class BaselineActor(nn.Module):
    def __init__(self, input_dim=12, use_mlp_actlayer=False) -> None:
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.base = MLPBase(input_dim, '128 128')
        self.rnn = GRULayer(128, 128, 1)
        self.act = ACTLayer(128, [41, 41, 41, 30], use_mlp_actlayer)
        self.to(torch.device('cpu'))

    def check(self, input):
        output = torch.from_numpy(input) if type(input) == np.ndarray else input
        return output

    def forward(self, obs, rnn_states):
        x = check(obs).to(**self.tpdv)
        h_s = check(rnn_states).to(**self.tpdv)
        x = self.base(x)
        x, h_s = self.rnn(x, h_s)
        actions = self.act(x)
        return actions, h_s
