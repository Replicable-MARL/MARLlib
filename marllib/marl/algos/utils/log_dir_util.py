import shutil
import os

__total, __used, __free = shutil.disk_usage(".")

available_local_dir = '{}/exp_results'.format(os.getcwd()) if __used / __total <= 0.95 else '/mnt/exp_results'