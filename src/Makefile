EIGEN = -I ../Dependencies/eigen/Eigen
GXX = g++

.PHONY : build clean
all: build

build: Convolutions.o CMatrix.o Input.o cNetwork.o init.cpp
	$(GXX) -g $(EIGEN) Convolutions.o CMatrix.o Input.o cNetwork.o init.cpp -o init.exe

Input.o: Input.cpp
	$(GXX) -g -c $(EIGEN) Input.cpp -o Input.o

cNetwork.o: CMatrix.o cNetwork.cpp
	$(GXX) -g -c CMatrix.o cNetwork.cpp -o cNetwork.o

CMatrix.o: Convolutions.o CMatrix.cpp
	$(GXX) -g -c Convolutions.o CMatrix.cpp -o CMatrix.o

Convolutions.o: Convolutions.cpp
	$(GXX) -g -c $(EIGEN) Convolutions.cpp -o Convolutions.o

clean:
	del init.exe
	del *.o