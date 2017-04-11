CXXFLAGS= -Ofast --std=c++11 -msse2 -march=native -I./lib -c
LD  = c++
LDFLAGS	=

test: test.o
	$(LD) $(LDFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ 

test.cpp: lstm_operations.h lstm_core.h

clean: 
	rm -f *.o test
