#!/bin/bash
mkdir -p build
cd build 
cmake .. && make -j12 
cd .. 
./workspace/pro