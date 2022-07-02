import shutil

__total, __used, __free = shutil.disk_usage(".")

available_local_dir = '~/ray_results' if __used / __total <= 0.95 else '/mnt/ray_results'