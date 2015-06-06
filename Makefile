all: build_pygco build_cython_irls

build_armadillo:
	cd pcaflow/extern; \
		make build_armadillo; \
		cd ../../

build_pygco:
	cd pcaflow/extern; \
		make build_pygco; \
		cd ../../

build_cython_irls:
	cd pcaflow/solver/cython; \
		make; \
		cd ../..
