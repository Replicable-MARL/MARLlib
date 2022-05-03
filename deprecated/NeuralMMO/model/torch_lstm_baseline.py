''' copied and modified from Neural-MMO RLlib_Wrapper '''

import gym
import trueskill

from torch.nn.utils import rnn
import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN

from neural_mmo.forge.ethyr.torch import policy
from neural_mmo.forge.ethyr.torch.policy import attention

from neural_mmo.forge.trinity import Env
from neural_mmo.forge.trinity.dataframe import DataType
from neural_mmo.forge.trinity.overlay import Overlay, OverlayRegistry

from NeuralMMO.model.utils.spaces import *
from NeuralMMO.model.utils.input import *
from NeuralMMO.model.utils.output import *


class Base(nn.Module):
    def __init__(self, config):
        '''Base class for baseline policies
        Args:
           config: A Configuration object
        '''
        super().__init__()
        self.embed = config.EMBED
        self.config = config

        self.output = Output(config)
        self.input = Input(config,
                           embeddings=policy.MixedDTypeInput,
                           attributes=policy.SelfAttention)

        self.valueF = nn.Linear(config.HIDDEN, 1)

    def hidden(self, obs, state=None, lens=None):
        '''Abstract method for hidden state processing, recurrent or otherwise,
        applied between the input and output modules
        Args:
           obs: An observation dictionary, provided by forward()
           state: The previous hidden state, only provided for recurrent nets
           lens: Trajectory segment lengths used to unflatten batched obs
        '''
        raise NotImplementedError('Implement this method in a subclass')

    def forward(self, obs, state=None, lens=None):
        '''Applies builtin IO and value function with user-defined hidden
        state subnetwork processing. Arguments are supplied by RLlib
        '''
        entityLookup = self.input(obs)
        hidden, state = self.hidden(entityLookup, state, lens)
        self.value = self.valueF(hidden).squeeze(1)
        actions = self.output(hidden, entityLookup)
        return actions, state


class Encoder(Base):
    def __init__(self, config):
        '''Simple baseline model with flat subnetworks'''
        super().__init__(config)
        h = config.HIDDEN

        self.ent = nn.Linear(2 * h, h)
        self.conv = nn.Conv2d(h, h, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(h * 6 * 6, h)

        self.proj = nn.Linear(2 * h, h)
        self.attend = policy.SelfAttention(self.embed, h)

    def hidden(self, obs, state=None, lens=None):
        # Attentional agent embedding
        agentEmb = obs['Entity']
        selfEmb = agentEmb[:, 0:1].expand_as(agentEmb)
        agents = torch.cat((selfEmb, agentEmb), dim=-1)
        agents = self.ent(agents)
        agents, _ = self.attend(agents)
        # agents = self.ent(selfEmb)

        # Convolutional tile embedding
        tiles = obs['Tile']
        self.attn = torch.norm(tiles, p=2, dim=-1)

        w = self.config.WINDOW
        batch = tiles.size(0)
        hidden = tiles.size(2)
        # Dims correct?
        tiles = tiles.reshape(batch, w, w, hidden).permute(0, 3, 1, 2)
        tiles = self.conv(tiles)
        tiles = self.pool(tiles)
        tiles = tiles.reshape(batch, -1)
        tiles = self.fc(tiles)

        hidden = torch.cat((agents, tiles), dim=-1)
        hidden = self.proj(hidden)
        return hidden, state


class BatchFirstLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batch_first=True, **kwargs)

    def forward(self, input, hx):
        h, c = hx
        h = h.transpose(0, 1)
        c = c.transpose(0, 1)
        hidden, hx = super().forward(input, [h, c])
        h, c = hx
        h = h.transpose(0, 1)
        c = c.transpose(0, 1)
        return hidden, [h, c]


class Recurrent(Encoder):
    def __init__(self, config):
        '''Recurrent baseline model'''
        super().__init__(config)
        self.lstm = BatchFirstLSTM(
            input_size=config.HIDDEN,
            hidden_size=config.HIDDEN)

    # Note: seemingly redundant transposes are required to convert between
    # Pytorch (seq_len, batch, hidden) <-> RLlib (batch, seq_len, hidden)
    def hidden(self, obs, state, lens):
        # Attentional input preprocessor and batching
        lens = lens.cpu() if type(lens) == torch.Tensor else lens
        hidden, _ = super().hidden(obs)
        config = self.config
        h, c = state

        TB = hidden.size(0)  # Padded batch of size (seq x batch)
        B = len(lens)  # Sequence fragment time length
        T = TB // B  # Trajectory batch size
        H = config.HIDDEN  # Hidden state size

        # Pack (batch x seq, hidden) -> (batch, seq, hidden)
        hidden = rnn.pack_padded_sequence(
            input=hidden.view(B, T, H),
            lengths=lens,
            enforce_sorted=False,
            batch_first=True)

        # Main recurrent network
        oldHidden = hidden
        hidden, state = self.lstm(hidden, state)
        newHidden = hidden

        # Unpack (batch, seq, hidden) -> (batch x seq, hidden)
        hidden, _ = rnn.pad_packed_sequence(
            sequence=hidden,
            batch_first=True,
            total_length=T)

        return hidden.reshape(TB, H), state


class NMMO_Baseline_LSTM(TorchRNN, nn.Module):
    '''Wrapper class for using our baseline models with RLlib'''

    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop('config')
        super().__init__(*args, **kwargs)
        nn.Module.__init__(self)

        self.space = actionSpace(self.config).spaces
        self.model = Recurrent(self.config)

    # Initial hidden state for RLlib Trainer
    def get_initial_state(self):
        return [self.model.valueF.weight.new(1, self.config.HIDDEN).zero_(),
                self.model.valueF.weight.new(1, self.config.HIDDEN).zero_()]

    def forward(self, input_dict, state, seq_lens):
        logitDict, state = self.model(input_dict['obs'], state, seq_lens)

        logits = []
        # Flatten structured logits for RLlib
        for atnKey, atn in sorted(self.space.items()):
            for argKey, arg in sorted(atn.spaces.items()):
                logits.append(logitDict[atnKey][argKey])

        return torch.cat(logits, dim=1), state

    def value_function(self):
        return self.model.value

    def attention(self):
        return self.model.attn
