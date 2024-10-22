from setuptools import find_packages, setup

setup(
    name='attention-sdf',
    package_dir={'': 'src'}
    packages=find_packages('src'),
    version='0.1.0',
    description='Attention based model for continuous SDF(signed distance function) prediction inspired by DeepSDF paper.',
    author='Ignacy Dąbkowski',
    license='',
)
