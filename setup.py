"""Setup configuration for the EGASN package."""

from setuptools import setup, find_packages

setup(
    name="egasn",
    version="0.1.0",
    author="Amit Kumar, Harsh Bhardwaj, Ritesh Kumar Pandit",
    author_email="23bsc12430@cuchd.in",
    description="Epipolar Graph-Attention Segmentation Network for Autonomous Vehicles",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=9.5.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
        ],
    },
)
