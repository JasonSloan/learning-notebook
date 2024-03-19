#! /bin/bash
cd build
cmake .. 
make
cd ..
cp build/mainproject workspace/
./workspace/mainproject