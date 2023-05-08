import os
import sys
import numpy as np
import math
import six
from gym import error



try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self):
        display = get_display(None)

        self.width = 600
        self.height = 600

        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.window.set_minimum_size(self.width, self.height)
        # self.window.set_maximum_size(500, 500)
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = os.path.dirname(__file__)
        self.fig_path = os.path.join(script_dir, "plot_save")

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def render(self, env, return_rgb_array=False):
        glClearColor(0, 0, 0, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._display_network(env)
        self._display_powerloss(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _display_network(self, env):
        env.res_pf_plot()
        img_net = pyglet.image.load(os.path.join(self.fig_path, "pf_res_plot.jpeg"))
        batch = pyglet.graphics.Batch()
        psp = pyglet.sprite.Sprite(img_net,
                                   0,
                                   0,
                                   batch=batch,    
                                )
        psp.scale_y = 0.8
        psp.scale_x = 0.86
        psp.update()
        batch.draw()

    def _display_powerloss(self, env):
        powerloss = env._get_res_line_loss().sum()
        label = pyglet.text.Label(
            f'The total power loss: \t{powerloss:.3f}',
            font_name="Times New Roman",
            font_size=24,
            bold=True,
            x=10,
            y=self.height-15,
            anchor_x="left",
            anchor_y="center",
            dpi=100
        )
        label.draw()
