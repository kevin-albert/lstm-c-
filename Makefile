CXX = g++-6
CXXFLAGS= -Ofast --std=c++17 -c -I./lib \
    -msse2 -march=native \
    -mfma -fopenmp \
    -DMKL_LP64 -I/opt/intel/mkl/include \
	-Wa,-q \
	-DEIGEN_USE_MKL_ALL -DEIGEN_FAST_MATH -DNDEBUG 

LD  = g++-6
LDFLAGS	= -llapack -lblas -fopenmp -mfma -framework Accelerate -Wa,-q \
	-Wl,-rpath,/opt/intel/mkl/lib \
	-Wl,-rpath,/opt/intel/compilers_and_libraries_2017.2.163/mac/compiler/lib \
	-L/opt/intel/mkl/lib \
	-L/opt/intel/compilers_and_libraries_2017.2.163/mac/compiler/lib \
	-lmkl_intel_lp64 \
 	-lmkl_intel_thread \
	-lmkl_core \
	-liomp5 \
	-lm -ldl 


# XXX for dev, compiles faster
#CXX = g++-6
#CXXFLAGS= -O2 --std=c++17 -c -I./lib
#LD  = g++-6
#LDFLAGS	= 


trump1: trump1.o lib/text_mapper.o lib/rng.o lib/lstm.o lib/read_data.o 
	$(LD) $(LDFLAGS) $^ -o $@

lib/%.o: lib/%.cpp lib/*.h
	$(CXX) $(CXXFLAGS) $< -o $@ 

%.o: %.cpp lib/*.h
	$(CXX) $(CXXFLAGS) $< -o $@ 
	
clean: 
	rm -f lib/*.o *.o trump1
