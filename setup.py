from setuptools import setup, find_packages
import codecs
import os.path


with open("README.md", "r") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(name='jtes',
      version=get_version("jtes/__init__.py"),
      packages=find_packages(),
      install_requires=["numpy"],
      author='Marc Wenninger',
      author_email='pypi@walwe.de',
      description='Implementation of the Jaccard Timespan Event Score',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
      ],
      python_requires='>=3.6',
      url='https://github.com/deddiag/python-jtes'
      )
