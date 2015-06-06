all : build_armadillo build_pygco
	
build_armadillo :
	cd armadillo; \
	echo ${PWD}; \
	cmake .; \
	make; \
	cd ..

build_pygco :
	rm pygco.so; \
	cd gco_python; \
	make; \
	cd ..; \
	ln -s gco_python/pygco.so . 

	
