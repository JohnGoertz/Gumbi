from setuptools import find_packages, setup
import pathlib as pl

VERSION = '0.1.1'
DISTNAME = "gumbi"
DESCRIPTION = "Gaussian Process Model Building Interface"
AUTHOR = "John Goertz"
AUTHOR_EMAIL = ""
URL = ""
LICENSE = ""

PROJECT_ROOT = pl.Path(__file__).resolve().parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
README = PROJECT_ROOT / "README.md"

with open(REQUIREMENTS) as f:
    install_reqs = f.read().splitlines()

with open(README, 'r') as fh:
    long_description = fh.read()

classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Operating System :: OS Independent",
]

setup(
    name=DISTNAME,
    version=VERSION,
    author="John Goertz",
    author_email="",
    description=DESCRIPTION,
    long_description=long_description,
    python_requires='>=3.9',
    packages=find_packages(),
    package_data={'': ['*.pkl', '*.mplstyle']},
    include_package_data=True,
    install_requires=install_reqs,
    classifiers=classifiers,
    #keywords=['python'],
)
