#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Gimme the version number that I should update the metadata to!"
    exit 1
fi

MAPELDIR=..

find "$MAPELDIR" -name 'pyproject.toml' |\
  xargs -I % sed -i '' -e 's\^version[[:blank:]]*=.*$\version = "'$1'"\g' %

echo "Version updated to "$1


