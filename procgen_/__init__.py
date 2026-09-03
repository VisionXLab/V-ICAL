import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
version_path = os.path.join(SCRIPT_DIR, "version.txt")
__version__ = open(version_path).read()

from .env import ProcgenEnv, ProcgenGym3Env
from .gym_registration import register_environments

register_environments()

import os,shutil,json
import subprocess as sp

conda_exe = shutil.which("conda")

if conda_exe is not None:
    try:
        conda_info = json.loads(
            sp.run(["conda", "info", "--json"], stdout=sp.PIPE).stdout
        )
    except FileNotFoundError:
        conda_info = json.loads(
            sp.run(["conda.bat", "info", "--json"], stdout=sp.PIPE).stdout
        )

    os.add_dll_directory(os.path.join(conda_info["active_prefix"],"library","bin"))
    os.add_dll_directory(os.path.join(conda_info["conda_prefix"],"library","bin"))

__all__ = ["ProcgenEnv", "ProcgenGym3Env"]
