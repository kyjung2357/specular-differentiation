"""Small build hooks used by the setuptools backend."""

from pathlib import Path
from shutil import rmtree

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class CleanBuildPy(_build_py):
    """Prevent deleted package modules from leaking out of an old build cache."""

    def run(self) -> None:
        source_package = (Path(__file__).resolve().parent / "specular").resolve()
        package_output = (Path(self.build_lib).resolve() / "specular")
        if package_output.resolve() == source_package:
            raise RuntimeError("refusing to clean the source package directory")
        if not self.dry_run and package_output.is_dir():
            rmtree(package_output)
        super().run()


setup(cmdclass={"build_py": CleanBuildPy})
