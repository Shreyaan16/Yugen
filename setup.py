from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="YuGen",
    version="2.0",
    author="Shreyaan16",
    packages=find_packages(),
    install_requires = requirements,
)