"""Setup script for the NameFrame package."""

from setuptools import find_packages, setup

setup(
    name="nameframe",
    version="0.1.0",
    description="A standardized, modular deep learning training framework",
    author="NameFrame contributors",
    python_requires=">=3.9",
    packages=find_packages(include=["nameframe", "nameframe.*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "munch>=4.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "psutil>=5.9.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "onnx": ["onnx", "onnxruntime"],
        "timm": ["timm>=0.9.0"],
        "albumentations": ["albumentations>=1.3.0"],
        "tensorboard": ["tensorboard>=2.13.0"],
        "gpu_monitor": ["nvidia-ml-py>=12.0.0"],
        "triton": ["triton>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "nameframe=nameframe.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
