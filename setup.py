#!/usr/bin/env python3
"""Setup script for CAASL package."""

from setuptools import find_packages, setup

setup(
    name="caasl",
    version="1.0.0",
    description="Causal Amortized Structure Learning with Deep Reinforcement Learning",
    author="Yashas Annadani",
    author_email="yashas.annadani@tum.de",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.14.5",
        "torch>=1.9.0",
        "torchvision",
        "gym>=0.21.0",
        "wandb",
        "igraph",
        "scipy",
        "setuptools",
        "dowel",
        "avici",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
            "pytest",
            "pytest-cov",
            "pre-commit",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
        ],
    },
    entry_points={
        "console_scripts": [
            "caasl=caasl.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
