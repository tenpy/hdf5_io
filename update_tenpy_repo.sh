#!/bin/bash
# This file copies files from this repo into (a local copy of) the TeNPy git repository (specified as argument).

set -e   # abort on error

if [[ $# -ne 1 ]] || [[ "$1" == "-h" ]] || [[ ! -f "$1/tenpy/__init__.py" ]]
then
	echo -e "usage: $0 tenpy_repo\n\nHere, 'tenpy_repo' points to a local tenpy git repository, to which the files from here should be copied."
	exit 1
fi

prompt_continue() {
	read -p "$1 " ANS
	if [[ "$ANS" != "y" ]]
	then
		exit 1
	fi
}

THISREPO="$(dirname "$(readlink -f "$0")")"
TENPYREPO="$1"

cd "$TENPYREPO"
if test -n "$(git status -s)" 
then
	echo "Erorr: git repository not clean; afraid to overwrite something."
	exit 1 
fi

cp "$THISREPO/src/python3/hdf5_io.py" "$TENPYREPO/tenpy/tools/io.py"
cp "$THISREPO/doc/input_output.rst" "$TENPYREPO/doc/intro/input_output.rst"
prompt_continue "Copied files. Continue [y/n]?"

git diff

prompt_continue "git commit these changes [y/n]?"

git add tenpy/tools/io.py
git add doc/intro/input_output.rst
git commit -m "Merged files from hdf5_io repository"
