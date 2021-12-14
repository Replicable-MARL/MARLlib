from run import *
from run_multi_trial import *

from config import *

if __name__ == '__main__':
    args = get_train_parser().parse_args()
    if args.parallel:
        run_parallel(args)
    else:
        run(args)

