# -*- coding: utf-8 -*-
"""Setup file."""
from setuptools import setup
from setuptools import find_packages

setup(name='plate',
      version="0.0.1",
      description='Plate composition',
      author='Stéphan Tulkens',
      author_email='stephan.tulkens@uantwerpen.be',
      url='https://github.com/stephantul/plate',
      license='GPLv3',
      packages=find_packages(),
      classifiers=[
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3'],
      keywords='machine learning',
      zip_safe=True,
      python_requires='>=3')
