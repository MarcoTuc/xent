# scrappy setup py to get started 

from setuptools import setup, find_packages
import os

work_dir = os.getcwd()
env_dir = os.path.join(work_dir, "menv")

setup(
    name="xent",
    version="0.1",
    packages=find_packages(where=env_dir),
)
