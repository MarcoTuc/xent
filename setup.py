from setuptools import setup, find_packages
import os

home_dir = os.path.expanduser("~")
work_dir = os.path.join(home_dir, "synth")
env_dir = os.path.join(work_dir, "menv")

# make necessary directories
try: os.makedirs(os.path.join(work_dir, "models"), exist_ok=False)
except OSError: pass 
try: os.makedirs(os.path.join(work_dir, "data"), exist_ok=False)
except OSError: pass

# make directories for highlighting folder -- will be removed once xent package is complete
hili_dir = os.path.join(work_dir, "highlighting")
try: os.makedirs(os.path.join(hili_dir, "models"), exist_ok=False)
except OSError: pass 
try: os.makedirs(os.path.join(hili_dir, "data"), exist_ok=False)
except OSError: pass

setup(
    name="xent",
    version="0.1",
    packages=find_packages(where=env_dir)
)