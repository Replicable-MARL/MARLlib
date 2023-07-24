# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import click
import os
import shutil
import subprocess

import ray


def do_link(file_path, force=False, local_path=None, packagent=None):
    file_path = os.path.abspath(os.path.join(packagent.__file__, f"../{file_path}"))

    # Infer local_path automatically.
    if local_path is None:
        local_path = f"../{file_path}"
    local_home = os.path.abspath(os.path.join(__file__, f"../{local_path}"))
    # If installed package dir does not exist, continue either way. We'll
    # remove it/create a link from there anyways.
    if not os.path.isfile(file_path):
        print(f"{file_path} does not exist. Continuing to link.")
    # Make sure the path we are linking to does exist.
    assert os.path.exists(local_home), local_home
    # Confirm with user.
    if not force and not click.confirm(
            f"This will replace:\n  {file_path}\nwith "
            f"a symlink to:\n  {local_home}",
            default=True):
        return

    # Windows: Create directory junction.
    if os.name == "nt":
        try:
            shutil.rmtree(file_path)
        except FileNotFoundError:
            pass
        except OSError:
            os.remove(file_path)

        # create symlink for directory or file
        if os.path.isdir(local_home):
            subprocess.check_call(
                ["mklink", "/J", file_path, local_home], shell=True)
        elif os.path.isfile(local_home):
            subprocess.check_call(
                ["mklink", "/H", file_path, local_home], shell=True)
        else:
            print(f"{local_home} is neither directory nor file. Link failed.")

    # Posix: Use `ln -s` to create softlink.
    else:
        sudo = []
        if not os.access(os.path.dirname(file_path), os.W_OK):
            print("You don't have write permission "
                  f"to {file_path}, using sudo:")
            sudo = ["sudo"]
        print(
            f"Creating symbolic link from \n {local_home} to \n {file_path}"
        )
        subprocess.check_call(sudo + ["rm", "-rf", file_path])
        subprocess.check_call(sudo + ["ln", "-s", local_home, file_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Setup dev.")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="RLlib_patch.")
    parser.add_argument(
        "--pommerman", "-p", action="store_true", help="pommerman.")
    args = parser.parse_args()

    do_link("rllib/execution/replay_buffer.py", force=args.yes, local_path="./rllib/execution/replay_buffer.py",
            packagent=ray)
    do_link("rllib/execution/train_ops.py", force=args.yes, local_path="./rllib/execution/train_ops.py", packagent=ray)

    # models
    do_link("rllib/models/preprocessors.py", force=args.yes, local_path="./rllib/models/preprocessors.py",
            packagent=ray)

    # policy
    do_link("rllib/policy/rnn_sequencing.py", force=args.yes, local_path="./rllib/policy/rnn_sequencing.py",
            packagent=ray)
    do_link("rllib/policy/torch_policy.py", force=args.yes, local_path="./rllib/policy/torch_policy.py", packagent=ray)

    # utils
    do_link("rllib/utils/exploration/ornstein_uhlenbeck_noise.py", force=args.yes,
            local_path="./rllib/utils/exploration/ornstein_uhlenbeck_noise.py", packagent=ray)

    if args.pommerman:
        import pommerman

        do_link('graphics.py', force=args.yes, local_path='pommerman/graphics.py', packagent=pommerman)

        do_link("__init__.py", force=args.yes, local_path='pommerman/__init__.py', packagent=pommerman)

        do_link("forward_model.py", force=args.yes, local_path="pommerman/forward_model.py",
                packagent=pommerman)

        do_link("envs/v0.py", force=args.yes, local_path="pommerman/v0.py", packagent=pommerman)

    print("finish soft link")
