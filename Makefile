CXX = g++-6
CXXFLAGS= -Ofast --std=c++17 -msse2 -march=native -I./lib -c \
	-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE \
	-DEIGEN_FAST_MATH \
	-fopenmp

LD  = g++-6
LDFLAGS	= -llapack -lblas -fopenmp

time: time.o
	$(LD) $(LDFLAGS) $^ -o $@


test: test.o
	$(LD) $(LDFLAGS) $^ -o $@

%.o: %.cpp *.h
	$(CXX) $(CXXFLAGS) $< -o $@ 

clean: 
	rm -f *.o test
