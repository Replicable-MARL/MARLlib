import numpy as np
import time
from mujoco_py import const, MjViewer, ignore_mujoco_warnings
import glfw
from gym.spaces import Box
from gym.spaces import MultiDiscrete


class EnvViewer(MjViewer):

    def __init__(self, env):
        self.env = env
        self.elapsed = [0]
        self.env.reset()
        self.seed = self.env.seed()
        super().__init__(self.env.unwrapped.sim)
        self.num_action = self.env.action_space.shape[0]
        self.action_mod_index = 0
        self.action = self.zero_action(self.env.action_space)

    def zero_action(self, action_space):
        if isinstance(action_space, Box):
            return np.zeros(action_space.shape[0])
        elif isinstance(action_space, MultiDiscrete):
            return action_space.nvec // 2  # assume middle element is "no action" action

    def env_reset(self):
        start = time.time()
        # get the seed before calling env.reset(), so we display the one
        # that was used for the reset.
        self.seed = self.env.seed()
        self.env.reset()
        self.elapsed.append(time.time() - start)
        self.update_sim(self.env.unwrapped.sim)

    def key_callback(self, window, key, scancode, action, mods):
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        if key == glfw.KEY_ESCAPE:
            self.env.close()

        # Increment experiment seed
        elif key == glfw.KEY_N:
            self.seed[0] += 1
            self.env.seed(self.seed)
            self.env_reset()
            self.action = self.zero_action(self.env.action_space)
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            self.seed = [max(self.seed[0] - 1, 0)]
            self.env.seed(self.seed)
            self.env_reset()
            self.action = self.zero_action(self.env.action_space)
        if key == glfw.KEY_A:
            if isinstance(self.env.action_space, Box):
                self.action[self.action_mod_index] -= 0.05
        elif key == glfw.KEY_Z:
            if isinstance(self.env.action_space, Box):
                self.action[self.action_mod_index] += 0.05
        elif key == glfw.KEY_K:
            self.action_mod_index = (self.action_mod_index + 1) % self.num_action
        elif key == glfw.KEY_J:
            self.action_mod_index = (self.action_mod_index - 1) % self.num_action

        super().key_callback(window, key, scancode, action, mods)

    def render(self):
        super().render()

        # Display applied external forces.
        self.vopt.flags[8] = 1

    def run(self, once=False):
        while True:
            with ignore_mujoco_warnings():
                self.env.step(self.action)
            self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
            self.add_overlay(const.GRID_TOPRIGHT, "Apply action", "A (-0.05) / Z (+0.05)")
            self.add_overlay(const.GRID_TOPRIGHT, "on action index %d out %d" % (self.action_mod_index, self.num_action), "J / K")
            self.add_overlay(const.GRID_BOTTOMRIGHT, "Reset took", "%.2f sec." % (sum(self.elapsed) / len(self.elapsed)))
            self.add_overlay(const.GRID_BOTTOMRIGHT, "Action", str(self.action))
            self.render()
            if once:
                return
