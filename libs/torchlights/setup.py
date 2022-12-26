from setuptools import setup, find_packages

setup(
    name='torchlights',
    packages=find_packages(),
    version='0.3.2',
    install_requires=['munch', 'colorama', 'readchar', 'tqdm', 'qqdm', 'pyyaml', 'colorlog'],
    include_package_data=True
)