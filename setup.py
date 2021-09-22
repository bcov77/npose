from setuptools import setup

__version__ = "1.0"

setup(
   name='npose',
   version='1.0',
   description='A lightning fast way to deal with pdb backbones',
   license='MIT',
   author='Brian Coventry',
   author_email='bcoventry77@gmail.com',
   url="https://github.com/bcov77/npose",
   download_url="https://github.com/bcov77/npose/archive/refs/tags/v1.0.tar.gz",
   packages=['npose_util', 'npose_util_pyrosetta', 'voxel_array'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
)
