# Usage:
# make or make all	# build package distribution
# make clean		# remove old package files 

.PHONY: all prereqs release readme clean help

all: prereqs build check

help:
	@echo "clean - remove all build, test and Python artifacts"
	@echo "prereqs - install required packages for building distribution"
	@echo "build - build package with setup.py"	
	@echo "check - run twine check on built distribution"
	@echo "readme - generate Markdown documentation LaTeX code with .svgs"
	@echo "release - publish packaged release to PyPI"
	@echo "venv - create and activate virtual environment with SciPy"

prereqs:
	python3 -m pip install --upgrade pip
	python3 -m pip install --user --upgrade setuptools wheel twine scipy numpy matplotlib

build: prereqs
	python3 -m setup sdist bdist_wheel 

check:
	twine check dist/*

release: build
	python3 -m twine upload dist/*

readme:
	python3 -m readme2tex --nocdn --readme read.tex.md --output README.md

venv:
	python3 -m venv venv
	source venv/bin/activate
	python3 -m pip install --upgrade numpy scipy matplotlib

clean:
	rm -r dist build __pycache__ *.egg-info *.swp *~ .*.un~ 
