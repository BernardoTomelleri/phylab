# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:29:11 2021

@author: berni
"""
import setuptools
import pathlib

from phylab import (__pkgname__ as PKG_NAME, __version__ as VERSION,
                       __author__ as AUTHOR, __license__ as LICENSE,
                       __summary__ as SUMMARY, __url__ as URL)

# The directory containing this file
cwd = pathlib.Path(__file__).parent

# The text of the README file
README = (cwd/'README.md').read_text()

setuptools.setup(
    name = PKG_NAME,
    version = VERSION,
    author  =  AUTHOR,
    author_email = 'berni.tomelleri@gmail.com',
    description = SUMMARY,
    long_description = README,
    long_description_content_type = 'text/markdown',
    url = URL,
    project_urls={
        "Bug Tracker": URL'/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0",
        "Operating System :: OS Independent",
    license = LICENSE,
    packages = setuptools.find_packages(),
    python_requires = '>=3.6',
    install_requires = [
        'numpy',
        'scipy',
        'matplotlib'
    ]
)