#!/bin/sh

set -xe
TAG=$(git describe --abbrev=0 --tags)
TAG=echo ${TAG} | tr -d "v"

sed -i "s/NEWVERSIONHERE/${TAG}/g" setup.py