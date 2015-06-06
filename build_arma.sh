#! /bin/bash

# Save current dir
BASEDIR=${PWD}
echo ${BASEDIR}

# Pretty formatting.
# See http://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

printf "${GREEN}    Building IRLS solver... \n${NC}"

cd pcaflow/solver/cython
rm RobustQuadraticSolverCython.so
python2 setup.py build_ext --inplace
cd ..
ln -fs cython/RobustQuadraticSolverCython.so .
cd ${BASEDIR}
printf "${GREEN}        done. \n${NC}"



