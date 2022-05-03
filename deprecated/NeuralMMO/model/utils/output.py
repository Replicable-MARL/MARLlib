''' copied and modified from Neural-MMO RLlib_Wrapper '''

from NeuralMMO.model.utils.input import *


class Output(nn.Module):
    def __init__(self, config):
        '''Network responsible for selecting actions
        Args:
           config: A Config object
        '''
        super().__init__()
        self.config = config
        self.h = config.HIDDEN

        self.net = DiscreteAction(self.config, self.h, self.h)
        self.arg = nn.Embedding(Action.n, self.h)

    def names(self, nameMap, args):
        '''Lookup argument indices from name mapping'''
        return np.array([nameMap.get(e) for e in args])

    def forward(self, obs, lookup):
        '''Populates an IO object with actions in-place

        Args:
           obs     : An IO object specifying observations
           lookup  : A fixed size representation of each entity
        '''
        rets = defaultdict(dict)
        for atn in Action.edges:
            for arg in atn.edges:
                lens = None
                if arg.argType == Fixed:
                    batch = obs.shape[0]
                    idxs = [e.idx for e in arg.edges]
                    cands = self.arg.weight[idxs]
                    cands = cands.repeat(batch, 1, 1)
                else:
                    cands = lookup['Entity']
                    lens = lookup['N']

                logits = self.net(obs, cands, lens)
                rets[atn][arg] = logits

        return rets
