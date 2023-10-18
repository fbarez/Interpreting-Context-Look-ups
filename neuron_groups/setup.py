from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='We explore neurons families in LLMs',
    author='fbarez',
    license='MIT',
    install_requires=[
        'transformer-lens',
        'numpy',
        'tiktoken',
    ]
)
