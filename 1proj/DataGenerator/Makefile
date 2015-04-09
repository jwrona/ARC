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

HDF5_DIR=/apps/libs/hdf5/intel/1.8.13_serial

CXX		= icpc

CXXFLAGS        = -std=c++11 -O3 -xavx -openmp -xhost  -I$(HDF5_DIR)/include -I.

TARGET		= arc_generator
LDFLAGS		= -std=c++11 -xavx -O3 -openmp -xhost  -L$(HDF5_DIR)/lib \
        -Wl,-rpath,$(HDF5_DIR)/lib


all:		$(TARGET)	


$(TARGET):	main.o 					

#---------------------------------------------------------

	$(CXX) $(LDFLAGS) main.o 			\
		-lhdf5_hl -lhdf5 \
		-o $@


$(TARGET).o : $(TARGET).cpp
	$(CXX) $(CXXFLAGS) -c $(TARGET).cpp

test:
	./arc_generator -o material.h5 -N 128 -H 100 -C 20


clean:
	rm -f *.o  $(TARGET)

