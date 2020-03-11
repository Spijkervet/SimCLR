#!/bin/sh
conda env create -f environment.yml || conda env update -f environment.yml || exit 1
