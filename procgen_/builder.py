import threading
import os
import contextlib
import subprocess as sp
import shutil
import json
import sys
import platform
import multiprocessing as mp
from pathlib import Path

import gym3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


global_build_lock = threading.Lock()
global_builds = set()


class RunFailure(Exception):
    pass


@contextlib.contextmanager
def nullcontext():
    # this is here for python 3.6 support
    yield


def run(cmd, cwd=None):
    return sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, encoding="utf8", cwd=cwd)


def check(proc, verbose):
    if proc.returncode != 0:
        print(f"RUN FAILED {proc.args}:\n{proc.stdout}")
        raise RunFailure("failed to build procgen from source")
    if verbose:
        print(f"RUN {proc.args}:\n{proc.stdout}")

def find_vs_generator():
    # Try to locate vswhere.exe
    vswhere_candidates = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    ]

    vswhere = None
    for p in vswhere_candidates:
        if p.exists():
            vswhere = str(p)
            break

    if vswhere is None:
        raise RuntimeError(
            "Could not find vswhere.exe. Install Visual Studio Build Tools "
            "or Visual Studio with the C++ workload."
        )

    cmd = [
        vswhere,
        "-products", "*",
        "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-format", "json",
    ]

    result = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True, check=True)
    installs = json.loads(result.stdout)

    if not installs:
        raise RuntimeError(
            "No Visual Studio installation with MSVC C++ tools was found. "
            "Install the 'Desktop development with C++' workload."
        )

    # Prefer newer VS versions
    installs.sort(
        key=lambda x: x.get("installationVersion", ""),
        reverse=True,
    )

    for inst in installs:
        version = inst.get("installationVersion", "")
        major = version.split(".")[0]

        if major == "18":
            return "Visual Studio 18 2026"
        elif major == "17":
            return "Visual Studio 17 2022"
        elif major == "16":
            return "Visual Studio 16 2019"

    raise RuntimeError(
        "Found Visual Studio, but not VS 2019 or VS 2022 or VS 2026."
    )

def _attempt_configure(build_type, package, build_type_dir):
    if "PROCGEN_CMAKE_PREFIX_PATH" in os.environ:
        cmake_prefix_paths = [os.environ["PROCGEN_CMAKE_PREFIX_PATH"]]
    else:
        # guess some common qt cmake paths, it's unclear why cmake can't find qt without this
        cmake_prefix_paths = ["/usr/local/opt/qt5/lib/cmake"]
        # Prefer the interpreter's environment: `conda info` can report the base
        # environment even when this process runs under a named environment.
        conda_prefix = sys.prefix
        if platform.system() == "Windows":
            conda_prefix = os.path.join(conda_prefix, "Library")
        cmake_prefix_paths.insert(0, os.path.join(conda_prefix, "lib", "cmake", "Qt5"))

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
            conda_prefix = conda_info["active_prefix"]
            if platform.system() == "Windows":
                conda_prefix = os.path.join(conda_prefix, "library")
            conda_cmake_path = os.path.join(conda_prefix, "lib", "cmake", "Qt5")
            cmake_prefix_paths.insert(0, conda_cmake_path)

            conda_prefix = conda_info["conda_prefix"]
            if platform.system() == "Windows":
                conda_prefix = os.path.join(conda_prefix, "library")
            conda_cmake_path = os.path.join(conda_prefix, "lib", "cmake", "Qt5")
            cmake_prefix_paths.insert(0, conda_cmake_path)

    generator = "Unix Makefiles"
    extra_configure_options = []
    if platform.system() == "Windows":
        generator = find_vs_generator()
        extra_configure_options.extend(["-A", "x64"])
    configure_cmd = [
        "cmake",
        "-G",
        generator,
        *extra_configure_options,
        "-DCMAKE_PREFIX_PATH=" + ";".join(cmake_prefix_paths),
        f"-DLIBENV_DIR={gym3.libenv.get_header_dir()}",
        SCRIPT_DIR,
    ]
    if package:
        configure_cmd.append("-DPROCGEN_PACKAGE=ON")
    if platform.system() != "Windows":
        # this is not used on windows, the option needs to be passed to cmake --build instead
        configure_cmd.append(f"-DCMAKE_BUILD_TYPE={build_type}")

    check(run(configure_cmd, cwd=build_type_dir), verbose=package)


def build(package=False, debug=False):
    """
    Build the requested environment in a process-safe manner and only once per process.
    """
    build_dir = os.path.join(SCRIPT_DIR, ".build")
    os.makedirs(build_dir, exist_ok=True)

    build_type = "relwithdebinfo"
    if debug:
        build_type = "debug"

    with global_build_lock:
        # check if we have built yet in this process
        if build_type not in global_builds:
            if package:
                # avoid the filelock dependency when building from setup.py
                lock_ctx = nullcontext()
            else:
                # prevent multiple processes from trying to build at the same time
                import filelock

                lock_ctx = filelock.FileLock(os.path.join(build_dir, ".build-lock"))
            with lock_ctx:
                sys.stdout.write("building procgen...")
                sys.stdout.flush()
                build_type_dir = os.path.join(build_dir, build_type)
                try:
                    os.makedirs(build_type_dir, exist_ok=True)
                    _attempt_configure(build_type, package, build_type_dir)
                except RunFailure:
                    # cmake can get into a weird state, so nuke the build directory and retry once
                    sys.stdout.write("retrying configure due to failure...")
                    sys.stdout.flush()
                    shutil.rmtree(build_type_dir)
                    os.makedirs(build_type_dir, exist_ok=True)
                    _attempt_configure(build_type, package, build_type_dir)

                if "MAKEFLAGS" not in os.environ:
                    os.environ["MAKEFLAGS"] = f"-j{mp.cpu_count()}"

                build_cmd = ["cmake", "--build", ".", "--config", build_type]
                check(run(build_cmd, cwd=build_type_dir), verbose=package)
                print("done")

            global_builds.add(build_type)

    lib_dir = os.path.join(build_dir, build_type)
    if platform.system() == "Windows":
        # the built library is in a different location on windows
        lib_dir = os.path.join(lib_dir, build_type)
    return lib_dir
