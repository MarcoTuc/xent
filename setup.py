# scrappy setup py to get started 

from setuptools import setup, find_packages
import os

work_dir = os.getcwd()
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

# requirements = []
# with open('requirements.txt', 'r') as file:
#     for line in file:
#         requirement = line.strip()
#         if requirement and not requirement.startswith('#'):
#             requirements.append(requirement)

setup(
    name="xent",
    version="0.1",
    packages=find_packages(where=env_dir),
)