from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "engine") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def run(self) -> None:
        if "editable_wheel" in sys.argv:
            return
        super().run()

    def build_extension(self, ext: CMakeExtension) -> None:
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = "Release"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        try:
            import pybind11

            cmake_args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
        except Exception:
            pass

        if "CMAKE_GENERATOR" not in os.environ:
            ninja = shutil.which("ninja")
            if ninja:
                cmake_args.extend(["-G", "Ninja"])

        subprocess.check_call(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", cfg, "--parallel"],
            cwd=build_temp,
        )

        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
        expected = extdir / f"{ext.name}{suffix}"
        if expected.exists():
            return

        matches = list(extdir.glob(f"{ext.name}*{suffix}"))
        if matches:
            shutil.copy2(matches[0], expected)
            return
        raise RuntimeError(f"CMake build did not produce {expected}")


def _editable_install() -> bool:
    return any(arg in {"editable_wheel", "develop"} for arg in sys.argv)


setup(
    ext_modules=[] if _editable_install() else [CMakeExtension("ptcg_engine")],
    cmdclass={"build_ext": CMakeBuild},
)
