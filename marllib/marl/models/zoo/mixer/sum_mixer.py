from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, states):
        return torch.sum(agent_qs, dim=2, keepdim=True)