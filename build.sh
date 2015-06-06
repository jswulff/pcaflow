#! /bin/bash

# Save current dir
BASEDIR=${PWD}
echo ${BASEDIR}

# Pretty formatting.
# See http://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Test for installed libraries

printf "${GREEN}Checking for libraries... \n${NC}"

names=( NumPy SciPy Cython Scikit-Learn OpenCV )
libs=( numpy scipy cython sklearn cv2 )
nlibs=$[${#names[@]}-1]
for i in $(seq 0 ${nlibs})
do
    n=${names[$i]}
    l=${libs[$i]}
    printf "${GREEN}    $n...${NC}"
    if python -c "import $l" > /dev/null
    then
        printf "${GREEN}Found. \n${NC}"
    else
        printf "${RED}\n *** Importing $n failed. Please install $n. *** \n${NC}"
        exit 1
    fi
done


# Check if libviso exists
if [ -e libviso2.zip ]
then
    printf "${GREEN}    Found libviso2.zip. Compiling python wrapper... \n${NC}"
    unzip -o libviso2.zip -d pcaflow/extern/libviso2/
    cd pcaflow/extern/libviso2/python
    rm libvisomatcher.so
    # Compile
    python2 setup.py build_ext --inplace
    cd ../../
    ln -sf libviso2/python/libvisomatcher.so .
    printf "${GREEN}    Finished compiling libviso2 python wrapper. \n${NC}"
else
    printf "${RED}\n Could not find libviso2.zip (see readme for details).\n"
    read -p "Do you want to use PCA-Flow with A-KAZE features? (Y/n)" -n 1 uselibviso
    printf "${NC}"
    if [ "${uselibviso}" == "n" ]
    then
        printf "\n${RED}Quiting...${NC}\n"
        cd ${BASEDIR}
        exit 1
    fi
fi

cd ${BASEDIR}

# Check for principal components
printf "${GREEN}Checking for principal components... ${NC}"

if [ -e data/ ] && \
    [ -e data/COV_KITTI.npy ] && \
    [ -e data/COV_SINTEL.npy ] && \
    [ -e data/COV_SINTEL_SUBLAYER.npy ] && \
    [ -e data/PC_U.npy ] && \
    [ -e data/PC_V.npy ]
then
    printf "${GREEN}Found. \n${NC}"
else
    printf "\n${RED}"
    read -p "Principal components not found. Download (filesize 484 MBytes)? (Y/n)" -n 1 downloadpc
    printf "${NC}"
    if [ "${downloadpc}" == "n" ]
    then
        printf "${RED}Continuing without download. Please provide your own principal components. \n${NC}"
    else
        printf "${GREEN}Downloading principal components... \n${NC}"
        curl http://files.is.tue.mpg.de/jwulff/pcaflow/principal_components.zip > principal_components.zip
        printf "${GREEN}Extracting principal components into data/... \n${NC}"
        mkdir -pv data/
        unzip -o principal_components.zip -d data/
        printf "${GREEN}Done.\n${NC}"
    fi
fi
    

# Check for internal libraries
printf "${GREEN}Building internal parts... \n${NC}"
printf "${GREEN}    Building pygco... \n${NC}"

cd pcaflow/extern
cd gco_python
rm pygco.so
make
cd ..
ln -sf gco_python/pygco.so .
cd ${BASEDIR}

printf "${GREEN}        done. \n${NC}"
printf "${GREEN}    Building IRLS solver... \n${NC}"

cd pcaflow/solver/cython
rm RobustQuadraticSolverCython.so
if [[ "$OSTYPE" == "linux"* ]]; then
    python2 setup_linux.py build_ext --inplace
elif [[ "$OSTYPE" == "darwin"* ]]; then
    python2 setup_osx.py build_ext --inplace
fi
cd ..
ln -fs cython/RobustQuadraticSolverCython.so .
cd ${BASEDIR}
printf "${GREEN}        done. \n${NC}"



