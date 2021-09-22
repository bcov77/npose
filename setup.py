from setuptools import setup

__version__ = "1.0"

setup(
   name='npose',
   version='1.0',
   description='npose is pretty chill',
   author='bcov',
   author_email='bcov@uw.edu',
   packages=['npose_util', 'npose_util_pyrosetta', 'voxel_array'],  #same as name
   install_requires=[], #external packages as dependencies
)
