#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Compiling the bare-metal C++ inference engine..."

# Compile with maximum optimization (-O3) and C++17 standard
g++ -O3 -std=c++17 infer.cpp -o infer

echo "Compilation complete! Executable 'infer' created."
echo "Run it using: ./infer"
