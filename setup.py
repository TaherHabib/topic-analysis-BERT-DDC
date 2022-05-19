from setuptools import find_packages, setup
from os import path
from pathlib import Path

ROOT = path.abspath(Path(__file__).parent)

# Get requirements from file
requirements = []
with open(path.join(ROOT, "requirements.txt")) as f:
    for line in f.readlines():
        requirements.append(line.strip())

setup(
    name='topic-analysis-bert',
    author='JohannesSchrumpf, TaherHabib',
    url='https://github.com/TaherHabib/topic-analysis-BERT-DDC',
    description='Using BERT to perform classification of books and courses according to Dewey-Decimal-Classification system',
    packages=find_packages(),
    install_requires=requirements,
    version='1.0'
)
