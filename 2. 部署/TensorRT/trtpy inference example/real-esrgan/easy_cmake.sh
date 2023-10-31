#!/bin/bash
cd build
rm -rf ./**
cmake ..
make
mv libinference.so ../lib/
