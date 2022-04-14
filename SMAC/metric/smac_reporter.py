from ray.tune import CLIReporter
from typing import Dict, List
from ray.tune.trial import Trial
from ray.rllib.utils.annotations import override


class SMACReporter(CLIReporter):

    @override(CLIReporter)
    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        print(self._progress_str(trials, done, *sys_info))


    """
    == Status ==
    Current time: 2022-04-14 00:22:21 (running for 00:14:29.50)
    Memory usage on this node: 10.0/15.5 GiB
    Using FIFO scheduling algorithm.
    Resources requested: 1.0/12 CPUs, 0.5/1 GPUs, 0.0/5.16 GiB heap, 0.0/2.58 GiB objects (0.0/1.0 accelerator_type:RTX)
    Result logdir: /home/username/ray_results/QMIX_GRU_3m
    Number of trials: 1/1 (1 RUNNING)
    +-------------------------------+----------+---------------------+
    | Trial name                    | status   | loc                 |
    |-------------------------------+----------+---------------------|
    | QMIX_grouped_smac_1b746_00000 | RUNNING  | 192.168.0.240:27912 |
    +-------------------------------+----------+---------------------+
    """