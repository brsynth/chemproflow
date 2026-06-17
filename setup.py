import glob

import setuptools
import yaml

# Version
version = ""
name = ""
fversion = glob.glob("**/_version.py", recursive=True)[0]
with open(fversion) as fid:
    lines = fid.read().splitlines()
    name = lines[0].split("=")[-1].strip().replace('"', "")
    version = lines[1].split("=")[-1].strip().replace('"', "")

# App name - dependencies
env = {}
with open("recipes/workflow.yaml") as fid:
    env = yaml.safe_load(fid)
install_requires = []
for package in env["dependencies"]:
    if isinstance(package, dict):
        raise ValueError("pip dependancies are defined in the yaml")
    if "=" in package:
        package = package.replace("=", "==")
    install_requires.append(package)
description = (
    "End-to-end mapping of membrane transport from chemical structure to microorganisms"
)

setuptools.setup(
    name=name,
    version=version,
    author=["guillaume-gricourt"],
    author_email=["guillaume.gricourt@inrae.fr"],
    description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/brsynth/chemproflow",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    entry_points={
        "console_scripts": ["chemproflow=chemproflow.pipeline.__main__:main"]
    },
)
