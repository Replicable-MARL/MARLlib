from ray.rllib.utils.framework import try_import_tf, try_import_torch
from marl.models.zoo.mixers import QMixer, VDNMixer
from marl.models.base.base_rnn import Base_RNN
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class VD_RNN(Base_RNN):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name, **kwargs)

        # mixer:
        if self.custom_config["global_state_flag"]:
            state_dim = self.custom_config["space_obs"]["state"].shape
        else:
            state_dim = self.custom_config["space_obs"]["obs"].shape + (self.custom_config["num_agents"], )
        if self.custom_config["algo_args"]["mixer"] == "qmix":
            self.mixer = QMixer(self.custom_config, state_dim)
        elif self.custom_config["algo_args"]["mixer"] == "vdn":
            self.mixer = VDNMixer()
        else:
            raise ValueError("Unknown mixer type {}".format(self.custom_config["algo_args"]["mixer"]))

    def mixing_value(self, all_agents_vf, state):
        # compatiable with rllib qmix mixer
        all_agents_vf = all_agents_vf.view(-1, 1, self.n_agents)
        v_tot = self.mixer(all_agents_vf, state)

        # shape to [B]
        return v_tot.flatten(start_dim=0)
