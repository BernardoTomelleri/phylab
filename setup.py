# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:29:11 2021

@author: berni
"""
from setuptools import setup, find_packages
from io import open
from os import path
import glob
import pathlib

from phylab import (__pkgname__ as PKG_NAME, __version__ as VERSION,
                       __author__ as AUTHOR, __license__ as LICENSE,
                       __summary__ as SUMMARY, __url__ as URL)

# The directory containing this file
PROJ_DIR = pathlib.Path(__file__).parent

# Get list of package requirements from .txt file
with open(path.join(PROJ_DIR, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')
REQUIREMENTS = [x.strip() for x in all_reqs]

# Load all python files in /examples dir as scripts
exmods = glob.glob(path.join(path.dirname(__file__), 'examples/*.py'))
EXAMPLES = [path.join('examples/', path.basename(f)) for f in exmods if path.isfile(f) and not f.endswith('__init__.py')]

# The text of the README file
README = (PROJ_DIR/'readpypi.md').read_text()

setup(
    name = PKG_NAME,
    version = VERSION,
    author  =  AUTHOR,
    author_email = 'berni.tomelleri@gmail.com',
    description = SUMMARY,
    long_description = README,
    long_description_content_type = 'text/markdown',
    url = URL,
    project_urls={
        'Bug Tracker': URL + '/issues'
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Education",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent"
    ],
    license = LICENSE,
    packages = find_packages(),
    scripts = EXAMPLES,
    python_requires = '>=3.6',
    install_requires = REQUIREMENTS
)