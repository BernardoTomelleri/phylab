# Usage:
# make or make all	# build package distribution
# make clean		# remove old package files 

.PHONY: all prereqs release readme clean 

all: prereqs build check

prereqs:
	python3 -m pip install --upgrade pip
	python3 -m pip install --user --upgrade setuptools wheels twine

build: prereqs
	python3 -m setup.py sdist bdist_wheel 

check:
	twine check dist/*

release: build
	python3 -m twine upload dist/*

readme:
	python3 -m readme2tex --nocdn --readme readtex.md --output README.md

clean:
	rm -r dist build __pycache__ *.egg-info *.swp *~ .*.un~ 
