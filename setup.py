from setuptools import setup, find_packages

setup(
    name="net_epistemology",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "networkx",
        "tqdm",
        "matplotlib",
        "seaborn",
        "dill",
    ],
)
