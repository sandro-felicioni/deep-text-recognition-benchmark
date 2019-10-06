from setuptools import setup

setup(
    name='text_extractor_deep_text_recognition',
    version='1.0.0',
    description='Deep Text Recognition module',
    packages=['text_extractor_deep_text_recognition'],
    install_requires=['lmdb', 'pillow', 'torchvision', 'nltk', 'natsort'],
)