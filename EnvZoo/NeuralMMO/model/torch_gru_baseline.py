''' copied and modified from Neural-MMO RLlib_Wrapper '''
from NeuralMMO.model.torch_lstm_baseline import *


class BatchFirstGRU(nn.GRU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batch_first=True, **kwargs)

    def forward(self, input, hx):
        h = hx[0]
        h = h.transpose(0, 1)
        hidden, h = super().forward(input, h)
        h = h.transpose(0, 1)
        return hidden, [h]


class Recurrent(Encoder):
    def __init__(self, config):
        '''Recurrent baseline model'''
        super().__init__(config)
        self.gru = BatchFirstGRU(
            input_size=config.HIDDEN,
            hidden_size=config.HIDDEN)

    # Note: seemingly redundant transposes are required to convert between
    # Pytorch (seq_len, batch, hidden) <-> RLlib (batch, seq_len, hidden)
    def hidden(self, obs, state, lens):
        # Attentional input preprocessor and batching
        lens = lens.cpu() if type(lens) == torch.Tensor else lens
        hidden, _ = super().hidden(obs)
        config = self.config

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
        hidden, state = self.gru(hidden, state)
        newHidden = hidden

        # Unpack (batch, seq, hidden) -> (batch x seq, hidden)
        hidden, _ = rnn.pad_packed_sequence(
            sequence=hidden,
            batch_first=True,
            total_length=T)

        return hidden.reshape(TB, H), state


class NMMO_Baseline_GRU(TorchRNN, nn.Module):
    '''Wrapper class for using our baseline models with RLlib'''

    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop('config')
        super().__init__(*args, **kwargs)
        nn.Module.__init__(self)

        self.space = actionSpace(self.config).spaces
        self.model = Recurrent(self.config)

    # Initial hidden state for RLlib Trainer
    def get_initial_state(self):
        return [self.model.valueF.weight.new(1, self.config.HIDDEN).zero_()]

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
