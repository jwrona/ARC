# /**
# * @File        Makefile
# * @Author      Jiri Jaros
# * @Affiliation FIT, Brno University of Technology
# * @Email       jarosjir@fit.vutbr.cz
# * @Comments    Linux makefile for Anselm
# * 
# * @Tool        ARC project 2015
# * @Created     10 April  2015, 10:49 AM
# * @LastModif   10 April  2015, 10:49 AM
#
# */


#Load following modules first for icpc
#   module load intel/15.2.164
#   module load impi
#   module load hdf5/1.8.13


#SET PATHS
#HDF5_DIR=/usr/local/hdf5-serial

HDF5_DIR=/apps/libs/hdf5/intel/1.8.13_parallel

#SET COMPILATOR, FLAGS and LIBS
CXX = mpicxx

CXXFLAGS=-W -Wall -Wextra -pedantic \
         -O3 -xavx -xhost \
         -std=c++11 \
         -I$(HDF5_DIR)/include -I.

LDFLAGS=-O3 -xavx \
        -std=c++11 \
        -L$(HDF5_DIR)/lib \
        -Wl,-rpath,$(HDF5_DIR)/lib


LIBS=-lhdf5 

TARGET=arc_proj02

all:		$(TARGET)	

$(TARGET): 	proj02.o \
		MaterialProperties.o \
		BasicRoutines.o \

	$(CXX) $(LDFLAGS) proj02.o MaterialProperties.o BasicRoutines.o\
     		$(LIBS)	\
	   	-o $@

$(TARGET).o : $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -c $(TARGET).cpp




test:
	./arc_proj02 -n 100 -m 0 -w 10 -i ../DataGenerator/material.h5 -o result.h5

	
	
clean:
	rm -f *.o
	rm -f *~
	rm -f $(TARGET)
