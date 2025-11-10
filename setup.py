"""
Setup script for LLMDTA
Note: Some packages may need manual installation (RDKit, ESM2)
"""

from setuptools import setup, find_packages

setup(
    name="llmdta",
    version="1.0.0",
    description="LLMDTA: Improving Cold-Start Prediction in Drug-Target Affinity with Biological LLM",
    author="Chris-Tang6",
    packages=find_packages(),
    install_requires=[
        "gensim==4.3.1",
        "matplotlib==3.2.2",
        "mol2vec==0.1",
        "numpy==1.23.4",
        "pandas==1.5.2",
        "scikit_learn==1.2.2",
        "scipy==1.8.1",
        "torch==1.8.2",
        "tqdm==4.65.0",
    ],
    python_requires=">=3.7,<3.10",
)


