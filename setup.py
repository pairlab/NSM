from setuptools import setup

requirements = [
    'numpy',
    'pyyaml',
    'trimesh',
    'pytorch_lightning',
    'tqdm',
    'yacs',
    'open3d'
]

exec(open("shape_assembly/version.py").read())

setup(
    name="shape_assembly",
    version="0.0.1",
    description="Code for Learning 3D Geometric Shape Assembly",
    long_description="Code for Learning 3D Geometric Shape Assembly",
    author="Yun-Chun Chen",
    author_email="ycchen@cs.toronto.edu",
    license="",
    url="",
    keywords="shape assembly",
    packages=["shape_assembly"],
    install_requires=requirements,
)
