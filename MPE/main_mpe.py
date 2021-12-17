from run_mpe import *
from run_mpe_multi_trial import *

from config_mpe import *

if __name__ == '__main__':
    args = get_train_parser().parse_args()
    if args.parallel:
        run_parallel(args)
    else:
        run(args)
