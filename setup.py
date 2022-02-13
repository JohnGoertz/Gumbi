from setuptools import find_packages, setup
import pathlib as pl

DISTNAME = "gumbi"
DESCRIPTION = "Gaussian Process Model Building Interface"
AUTHOR = "John Goertz"
AUTHOR_EMAIL = ""
URL = "https://github.com/JohnGoertz/Gumbi"
LICENSE = "Apache 2.0"

PROJECT_ROOT = pl.Path(__file__).resolve().parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
README = PROJECT_ROOT / "README.md"
VERSION = PROJECT_ROOT / "VERSION"

with open(REQUIREMENTS) as f:
    install_reqs = f.read().splitlines()

with open(README, 'r') as fh:
    long_description = fh.read()

with open(VERSION, encoding="utf-8") as f:
    version = f.read().strip()

classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
]

setup(
    name=DISTNAME,
    version=version,
    author="John Goertz",
    author_email="",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url=URL,
    license=LICENSE,
    python_requires='>=3.9',
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_reqs,
    classifiers=classifiers,
    #keywords=['python'],
)
