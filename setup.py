import sys
import platform
from pathlib import Path

from setuptools import find_packages, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


def _fetch_requirements(path: Path) -> list[str]:
    with path.open() as fd:
        requirements: list[str] = []

        for line in fd.readlines():
            requirement = line.split("#", maxsplit=1)[0].strip()
            if requirement:
                requirements.append(requirement)

        return requirements


# Custom wheel class to modify the wheel name
class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi_tag = f"{python_version}"

        if platform.system() == "Linux":
            platform_tag = "manylinux1_x86_64"
        else:
            platform_tag = platform.system().lower()

        return python_version, abi_tag, platform_tag


# Setup configuration
setup(
    author="miles Team",
    name="miles",
    version="0.2.1",
    packages=find_packages(include=["miles*", "miles_plugins*"]),
    include_package_data=True,
    install_requires=_fetch_requirements(Path("requirements.txt")),
    extras_require={
        "fsdp": [
            "torch>=2.0",
        ]
    },
    python_requires=">=3.10,<3.13",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    cmdclass={"bdist_wheel": bdist_wheel},
)
