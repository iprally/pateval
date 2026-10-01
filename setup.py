import re

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as readme:
    long_description = readme.read()
with open("pateval/__init__.py", encoding="utf-8") as init:
    version = re.search(r'^__version__ = "([^"]+)"', init.read(), re.MULTILINE).group(1)

setup(
    name="pateval",
    version=version,
    description="Evaluation harness for patent retrieval benchmarks, with the reading budget as a first-class variable.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nikolai Zenovkin",
    author_email="nikolai.zenovkin@iprally.com",
    url="https://github.com/iprally/pateval",
    license="MIT",
    license_files=["LICENSE"],
    packages=find_packages(include=["pateval", "pateval.*"]),
    package_data={"pateval": ["py.typed"]},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pytrec-eval-terrier>=0.5",
        "pyarrow>=14",
        "huggingface-hub>=0.24",
    ],
    extras_require={
        # Encoders for public Hugging Face models. The core package scores vectors and never imports torch.
        # torch 2.6 is the first whose torch.load defaults to weights only; some registry models ship only
        # pytorch_model.bin, which transformers reads with torch.load.
        "hf": ["torch>=2.6", "transformers>=4.40", "safetensors>=0.4"],
        "dev": ["pytest>=8.0"],
    },
    entry_points={"console_scripts": ["pateval = pateval.cli:main"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
