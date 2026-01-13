"""
Setup script for Neural Machine Translation System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('-'):
            requirements.append(line)

setup(
    name="neural-machine-translation",
    version="1.0.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="A modern neural machine translation system with state-of-the-art transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural-machine-translation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch-audio>=2.0.0",
            "torch-vision>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nmt-cli=cli:main",
            "nmt-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="machine-translation, nlp, transformers, huggingface, neural-networks",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/neural-machine-translation/issues",
        "Source": "https://github.com/yourusername/neural-machine-translation",
        "Documentation": "https://github.com/yourusername/neural-machine-translation#readme",
    },
)
