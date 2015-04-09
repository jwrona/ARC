# /**
# * @File        Makefile
# * @Author      Jiri Jaros
# * @Affiliation FIT, Brno University of Technology
# * @Email       jarosjir@fit.vutbr.cz
# * @Comments    Linux makefile for Anselm
# * 
# * @Tool        ARC project 2015
# * @Created     15 February  2015, 15:25 AM
# * @LastModif   12 March     2015, 07:47 AM
#
# */


#Load following modules first for GCC
#   module load gcc/4.9.0
#   module load hdf5/1.8.13

#Load following modules first for icpc
#   module load intel/14.0.1
#   module load hdf5/1.8.13


#SET PATHS
#HDF5_DIR=/usr/local/hdf5-serial

HDF5_DIR=/apps/libs/hdf5/intel/1.8.13_serial

#SET COMPILATOR, FLAGS and LIBS
CXX = icpc

CXXFLAGS=-W -Wall -Wextra -pedantic \
         -O3 -xavx -xhost -openmp \
         -std=c++11 \
         -I$(HDF5_DIR)/include -I.

LDFLAGS=-O3 -xavx -openmp \
        -std=c++11 \
        -L$(HDF5_DIR)/lib \
        -Wl,-rpath,$(HDF5_DIR)/lib


LIBS=-lhdf5 

TARGET=arc_proj01

all:		$(TARGET)	

$(TARGET): 	proj01.o \
		MaterialProperties.o \
		BasicRoutines.o \

	$(CXX) $(LDFLAGS) proj01.o MaterialProperties.o BasicRoutines.o\
     		$(LIBS)	\
	   	-o $@

$(TARGET).o : $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -c $(TARGET).cpp




test:
	./arc_proj01 -n 100 -m 0 -w 10 -t 1 -i ../DataGenerator/material.h5 -o result.h5

	
	
clean:
	rm -f *.o
	rm -f *~
	rm -f $(TARGET)
