from mujoco_worldgen.util.envs.flexible_load import load_env
from mujoco_worldgen.util.envs.env_viewer import EnvViewer


def examine_env(env_name, env_kwargs, core_dir, envs_dir, xmls_dir='xmls',
                env_viewer=EnvViewer, seed=None):
    '''
        Loads an environment and allows the user to examine it.
        Args:
            env_name (str): Environment name. Does not need to be exact since the load_env function
                does a search over the environments & xmls folder for any file names that match
                the env_name pattern
            env_kwargs (dict): Dictionary of environment keyword arguments
            core_dir (str): Absolute path to the core code directory for the project containing the
                environments we want to examine. This is usually the top-level git repository
                folder - in the case of the mujoco-worldgen repo, it would be the 'mujoco-worldgen'
                folder.
            envs_dir (str): relative path (from core_dir) to folder containing all environment files.
            xmls_dir (str): relative path (from core_dir) to folder containing all xml files.
            env_viewer (class): class used to render the environment. See the imported EnvViewer
                class for an example of how to structure this.
            seed (int): Environment seed
    '''
    env, args_remaining = load_env(env_name,
                                   core_dir=core_dir, envs_dir=envs_dir, xmls_dir=xmls_dir,
                                   return_args_remaining=True, **env_kwargs)
    if seed is not None:
        env.seed(seed)
    assert len(args_remaining) == 0, (
        f"There left unused arguments: {args_remaining}. There shouldn't be any.")
    if env is not None:
        env_viewer(env).run()
    else:
        print('"{}" doesn\'t seem to be a valid environment'.format(env_name))

    print("Error couldn't match against any of patterns. Please try to be more verbose.")
    print("\n\nFailed to examine")
