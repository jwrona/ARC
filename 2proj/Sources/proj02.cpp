/**
 * @file        proj02.cpp
 * @author      Jiri Jaros and Radek Hrbacek\n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       Parallelisation of Heat Distribution Method in Heterogenous
 *              Media using OpenMP
 *
 * @version     2015
 * @date        10 April 2015, 10:22 (created) \n
 * @date        10 April 2015, 10:22 (last revised) \n
 *
 * @detail
 * This is the main file of the project. Add all code here.
 */


#include <mpi.h>

#include <string.h>
#include <string>
#include <cmath>

#include <hdf5.h>

#include <sstream>
#include <immintrin.h>

#include "MaterialProperties.h"
#include "BasicRoutines.h"


using namespace std;


//----------------------------------------------------------------------------//
//---------------------------- Global variables ------------------------------//
//----------------------------------------------------------------------------//

/// Temperature data for sequential version.
float *seqResult = NULL;
/// Temperature data for parallel method.
float *parResult = NULL;

/// Parameters of the simulation
TParameters parameters;

/// Material properties
TMaterialProperties materialProperties;


//----------------------------------------------------------------------------//
//------------------------- Function declarations ----------------------------//
//----------------------------------------------------------------------------//

/// Sequential implementation of the Heat distribution
void SequentialHeatDistribution(float                     *seqResult,
                                const TMaterialProperties &materialProperties,
                                const TParameters         &parameters,
                                string                     outputFileName);

/// Parallel Implementation of the Heat distribution (Non-overlapped file output)
void ParallelHeatDistribution(float                     *parResult,
                              const TMaterialProperties &materialProperties,
                              const TParameters         &parameters,
                              string                     outputFileName);

/// Store time step into output file
void StoreDataIntoFile(hid_t         h5fileId,
                       const float * data,
                       const size_t  edgeSize,
                       const size_t  snapshotId,
                       const size_t  iteration);


//----------------------------------------------------------------------------//
//------------------------- Function implementation  -------------------------//
//----------------------------------------------------------------------------//


void ComputePoint(float  *oldTemp,
                  float  *newTemp,
                  float  *params,
                  int    *map,
                  size_t  i,
                  size_t  j,
                  size_t  edgeSize,
                  float   airFlowRate,
                  float   coolerTemp)
{
  // [i] Calculate neighbor indices
  const int center = i * edgeSize + j;
  const int top    = center - edgeSize;
  const int bottom = center + edgeSize;
  const int left   = center - 1;
  const int right  = center + 1;

  // [ii] The reciprocal value of the sum of domain parameters for normalization
  const float frac = 1.0f / (params[top]    +
                             params[bottom] +
                             params[left]   +
                             params[center] +
                             params[right]);

  // [iii] Calculate new temperature in the grid point
  float pointTemp = 
        oldTemp[top]    * params[top]    * frac +
        oldTemp[bottom] * params[bottom] * frac +
        oldTemp[left]   * params[left]   * frac +
        oldTemp[right]  * params[right]  * frac +
        oldTemp[center] * params[center] * frac;

  // [iv] Remove some of the heat due to air flow (5% of the new air)
  pointTemp = (map[center] == 0)
              ? (airFlowRate * coolerTemp) + ((1.0f - airFlowRate) * pointTemp)
              : pointTemp;

  newTemp[center] = pointTemp;
}

/**
 * Sequential version of the Heat distribution in heterogenous 2D medium
 * @param [out] seqResult          - Final heat distribution
 * @param [in]  materialProperties - Material properties
 * @param [in]  parameters         - parameters of the simulation
 * @param [in]  outputFileName     - Output file name (if NULL string, do not store)
 *
 */
void SequentialHeatDistribution(float                      *seqResult,
                                const TMaterialProperties &materialProperties,
                                const TParameters         &parameters,
                                string                     outputFileName)
{
  // [1] Create a new output hdf5 file
  hid_t file_id = H5I_INVALID_HID;
  
  if (outputFileName != "")
  {
    if (outputFileName.find(".h5") == string::npos)
      outputFileName.append("_seq.h5");
    else
      outputFileName.insert(outputFileName.find_last_of("."), "_seq");
    
    file_id = H5Fcreate(outputFileName.c_str(),
                        H5F_ACC_TRUNC,
                        H5P_DEFAULT,
                        H5P_DEFAULT);
    if (file_id < 0) ios::failure("Cannot create output file");
  }


  // [2] A temporary array is needed to prevent mixing of data form step t and t+1
  float *tempArray = (float *)_mm_malloc(materialProperties.nGridPoints * 
                                         sizeof(float), DATA_ALIGNMENT);

  // [3] Init arrays
  for (size_t i = 0; i < materialProperties.nGridPoints; i++)
  {
    tempArray[i] = materialProperties.initTemp[i];
    seqResult[i] = materialProperties.initTemp[i];
  }

  // [4] t+1 values, t values
  float *newTemp = seqResult;
  float *oldTemp = tempArray;

  if (!parameters.batchMode) 
    printf("Starting sequential simulation... \n");
  
  //---------------------- [5] start the stop watch ------------------------------//
  double elapsedTime = MPI_Wtime();
  size_t i, j;
  size_t iteration;
  float middleColAvgTemp = 0.0f;

  /* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
  //for (int i = 0; i < materialProperties.nGridPoints; ++i) {
  //  newTemp[i] = oldTemp[i] = i;
  //  materialProperties.domainMap[i] = i;
  //  materialProperties.domainParams[i] = i;
  //}
  for (int i = 0; i < materialProperties.edgeSize; ++i) {
    for (int j = 0; j < materialProperties.edgeSize; ++j) {
      printf("%3.3f ", materialProperties.domainParams[i * materialProperties.edgeSize + j]);
    }
    putchar('\n');
  }
  /* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

  // [6] Start the iterative simulation
  for (iteration = 0; iteration < parameters.nIterations; iteration++)
  {
    // [a] calculate one iteration of the heat distribution (skip the grid points at the edges)
    for (i = 1; i < materialProperties.edgeSize - 1; i++)
      for (j = 1; j < materialProperties.edgeSize - 1; j++)
        ComputePoint(oldTemp,
                     newTemp,
                     materialProperties.domainParams,
                     materialProperties.domainMap,
                     i, j,
                     materialProperties.edgeSize, 
                     parameters.airFlowRate,
                     materialProperties.coolerTemp);

    // [b] Compute the average temperature in the middle column
    middleColAvgTemp = 0.0f;
    for (i = 0; i < materialProperties.edgeSize; i++)
      middleColAvgTemp += newTemp[i*materialProperties.edgeSize +
                          materialProperties.edgeSize/2];
    middleColAvgTemp /= materialProperties.edgeSize;

    // [c] Store time step in the output file if necessary
    if ((file_id != H5I_INVALID_HID)  && ((iteration % parameters.diskWriteIntensity) == 0))
    {
      StoreDataIntoFile(file_id,
                        newTemp,
                        materialProperties.edgeSize,
                        iteration / parameters.diskWriteIntensity,
                        iteration);
    }

    // [d] Swap new and old values
    swap(newTemp, oldTemp);

    // [e] Print progress and average temperature of the middle column
    //if ((iteration % (parameters.nIterations / 10l)) == 
    //    ((parameters.nIterations / 10l) - 1l) && !parameters.batchMode)
    //{
    //  printf("Progress %ld%% (Average Temperature %.2f degrees)\n", 
    //         iteration / (parameters.nIterations / 100) + 1, 
    //         middleColAvgTemp);
    //}
  } // for iteration

  /* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
  for (int i = 0; i < materialProperties.edgeSize; ++i) {
    for (int j = 0; j < materialProperties.edgeSize; ++j) {
      printf("%3.0f ", oldTemp[i * materialProperties.edgeSize + j]);
    }
    putchar('\n');
  }
  /* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

  //-------------------- stop the stop watch  --------------------------------//  
  double totalTime = MPI_Wtime() - elapsedTime;

  // [7] Print final result
  if (!parameters.batchMode)
    printf("\nExecution time of sequential version %.5f\n", totalTime);
  else
    printf("%s;%s;%f;%e;%e\n", outputFileName.c_str(), "seq",
                               middleColAvgTemp, totalTime,
                               totalTime / parameters.nIterations);   

  // Close the output file
  if (file_id != H5I_INVALID_HID) H5Fclose(file_id);

  // [8] Return correct results in the correct array
  if (iteration & 1)
    memcpy(seqResult, tempArray, materialProperties.nGridPoints * sizeof(float));

  _mm_free(tempArray);
} // end of SequentialHeatDistribution
//------------------------------------------------------------------------------


/**
 * Parallel version of the Heat distribution in heterogenous 2D medium
 * @param [out] parResult          - Final heat distribution
 * @param [in]  materialProperties - Material properties
 * @param [in]  parameters         - parameters of the simulation
 * @param [in]  outputFileName     - Output file name (if NULL string, do not store)
 *
 * @note This is the function that students should implement.                                                  
 */
void ParallelHeatDistribution(float                     *parResult,
                              const TMaterialProperties &materialProperties,
                              const TParameters         &parameters,
                              string                     outputFileName)
{
  //--------------------------------------------------------------------------//
  //---------------- THE SECTION WHERE STUDENTS MAY ADD CODE -----------------//
  //--------------------------------------------------------------------------//
#define ROOT_PROC 0
#define TAG 0

#define DIM_X 0
#define DIM_Y 1
#define NDIMS 2

#define LEFT  0
#define RIGHT 1
#define UPPER 2
#define LOWER 3
#define NHALOS     4

  const int world_rank = MPI::COMM_WORLD.Get_rank();
  const int world_size = MPI::COMM_WORLD.Get_size();

  struct {
    std::size_t size[NDIMS]; //one tile size
    std::size_t data_size[NDIMS]; //actual data dimensions

    float *data_old, *data_new; //one tile t and t + 1 data
    std::size_t halo_send_offset[NHALOS]; //halo zones offsets
    std::size_t halo_recv_offset[NHALOS]; //halo zones offsets

    int *map; //without halo zones
    float *params; //with halo zones
  } tile;

  /* Create a new output hdf5 file. Root processor only. */
  hid_t file_id = H5I_INVALID_HID;
  if (world_rank == ROOT_PROC) {
    if (outputFileName != "")
    {
        if (outputFileName.find(".h5") == string::npos) {
            outputFileName.append("_par.h5");
        } else {
            outputFileName.insert(outputFileName.find_last_of("."), "_par");
        }

        file_id = H5Fcreate(outputFileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
		H5P_DEFAULT);
        if (file_id < 0) ios::failure("Cannot create output file");
    }
  }

  /* Partition input data grid into smaller grids - tiles. All processors. */
  std::size_t tiles_per[NDIMS]; //tiles per one dimension
  tiles_per[DIM_X] = std::sqrt(world_size / (1 + (static_cast<std::size_t>(
					  std::log2(world_size))& 1)));
  tiles_per[DIM_Y] = world_size / tiles_per[DIM_X];

  /* Fill tile struct: calculate tile dimensions, allocate memory for step t
   * and t + 1, calculate halo zones offsets, allocate memory for tile map and
   * tile parameters. All processors.
   */
  tile.size[DIM_X] = materialProperties.edgeSize / tiles_per[DIM_X];
  tile.size[DIM_Y] = materialProperties.edgeSize / tiles_per[DIM_Y];
  tile.data_size[DIM_X] = tile.size[DIM_X] + 2; //for halo zones
  tile.data_size[DIM_Y] = tile.size[DIM_Y] + 2; //for halo zones
  tile.data_old = static_cast<float *>(_mm_malloc(tile.data_size[DIM_X] *
			  tile.data_size[DIM_Y] * sizeof(float), DATA_ALIGNMENT));
  tile.data_new = static_cast<float *>(_mm_malloc(tile.data_size[DIM_X] *
			  tile.data_size[DIM_Y] * sizeof(float), DATA_ALIGNMENT));
  tile.halo_send_offset[LEFT] = tile.data_size[DIM_X] + 1;
  tile.halo_send_offset[RIGHT] = tile.data_size[DIM_X] * 2 - 2;
  tile.halo_send_offset[UPPER] = tile.data_size[DIM_X] + 1;
  tile.halo_send_offset[LOWER] = tile.data_size[DIM_X] * (tile.data_size[DIM_Y] - 2) + 1;

  tile.halo_recv_offset[LEFT] = tile.data_size[DIM_X];
  tile.halo_recv_offset[RIGHT] = tile.data_size[DIM_X] * 2 - 1;
  tile.halo_recv_offset[UPPER] = 1;
  tile.halo_recv_offset[LOWER] = tile.data_size[DIM_X] * (tile.data_size[DIM_Y] - 1) + 1;

  tile.map = static_cast<int *>(_mm_malloc(tile.data_size[DIM_X] *
			  tile.data_size[DIM_Y] * sizeof(int), DATA_ALIGNMENT));
  tile.params = static_cast<float *>(_mm_malloc(tile.data_size[DIM_X] *
			  tile.data_size[DIM_Y] * sizeof(float), DATA_ALIGNMENT));

  /* Create a new communicator - 2D grid Cartesian topology. All processors. */
  const int dims[NDIMS] = {static_cast<const int>(tiles_per[DIM_X]),
	  static_cast<const int>(tiles_per[DIM_Y])};
  const bool periods[NDIMS] = {false, false};
  auto grid_comm = MPI::COMM_WORLD.Create_cart(NDIMS, dims, periods, true);
  const int grid_rank = grid_comm.Get_rank(), grid_size = grid_comm.Get_size();

  /* Create a new MPI data type - subarray corresponding to one tile
   * among whole data board. All processors.
   */
  int sizes[NDIMS] = {materialProperties.edgeSize, materialProperties.edgeSize};
  int subsizes[NDIMS] = {tile.size[DIM_Y], tile.size[DIM_X]}; //rows, then cols
  int starts[NDIMS] = {0, 0};
  MPI::Datatype mpi_board_t = MPI::FLOAT.Create_subarray(NDIMS, sizes, subsizes,
		  starts, MPI::ORDER_C);
  mpi_board_t = mpi_board_t.Create_resized(0, tile.size[DIM_X] * sizeof(float));
  mpi_board_t.Commit();

  /* Create a new MPI data type - subarray corresponding to one tile
   * in array with data and halo zones. All processors. */
  sizes[0] = tile.data_size[DIM_Y]; //rows first
  sizes[1] = tile.data_size[DIM_X]; //cols second
  starts[0] = starts[1] = 1; //leave halo zones untouched
  MPI::Datatype mpi_tile_t = MPI::FLOAT.Create_subarray(NDIMS, sizes, subsizes,
		  starts, MPI::ORDER_C);
  mpi_tile_t.Commit();
  MPI::Datatype mpi_tile_map_t = MPI::INT.Create_subarray(NDIMS, sizes, subsizes,
		  starts, MPI::ORDER_C);
  mpi_tile_map_t.Commit();

  /* Calculate displacement in data board for each tile. Root processor only. */
  int sendcounts[grid_size], displs[grid_size];
  if (grid_rank == ROOT_PROC) {
    for (int rank = 0; rank < grid_size; ++rank) {
      int rank_coords[NDIMS];

      grid_comm.Get_coords(rank, NDIMS, rank_coords);
      sendcounts[rank] = 1;
      displs[rank] = rank_coords[0] + rank_coords[1] * tiles_per[DIM_X] *
	      tile.size[DIM_Y];
    }
  }

  /* Scatter initial temperature, domain map and parameters into tiles. */
  //for (int i = 0; i < materialProperties.nGridPoints; ++i) {
  //  materialProperties.initTemp[i] = i;
  //  materialProperties.domainMap[i] = i;
  //  materialProperties.domainParams[i] = i;
  //}
  grid_comm.Scatterv(materialProperties.initTemp, sendcounts, displs,
		  mpi_board_t, tile.data_old, 1, mpi_tile_t, ROOT_PROC);
  grid_comm.Scatterv(materialProperties.domainMap, sendcounts, displs,
		  mpi_board_t, tile.map, 1, mpi_tile_map_t, ROOT_PROC);
  grid_comm.Scatterv(materialProperties.domainParams, sendcounts, displs,
		  mpi_board_t, tile.params, 1, mpi_tile_t, ROOT_PROC);

  /* Create row and column halo data types. All processors. */
  MPI::Datatype mpi_halo_t[NDIMS];
  mpi_halo_t[DIM_X] = MPI::FLOAT.Create_vector(tile.size[DIM_Y], 1,
		  tile.data_size[DIM_X]); //column halo
  mpi_halo_t[DIM_Y] = MPI::FLOAT.Create_vector(1, tile.size[DIM_X],
		  tile.data_size[DIM_X]); //row halo
  mpi_halo_t[DIM_X].Commit();
  mpi_halo_t[DIM_Y].Commit();

  int neighbor_rank[NDIMS * NDIMS];
  for (std::size_t dim = 0, index = 0; dim < NDIMS; dim++) { //X, Y
    int dummy;

    grid_comm.Shift(dim, -1, dummy, neighbor_rank[index++]); //left/up
    grid_comm.Shift(dim, 1, dummy, neighbor_rank[index++]); //right/down
  }

  /* Initiation - send and reveive halo zones. All processors. */
  for (std::size_t i = 0; i < NDIMS * NDIMS; ++i) {
    grid_comm.Isend(tile.data_old + tile.halo_send_offset[i], 1,
		    mpi_halo_t[i / 2], neighbor_rank[i], TAG);
    grid_comm.Isend(tile.params + tile.halo_send_offset[i], 1,
		    mpi_halo_t[i / 2], neighbor_rank[i], TAG);
  }
  for (std::size_t i = 0; i < NDIMS * NDIMS; ++i) {
    grid_comm.Recv(tile.data_old + tile.halo_recv_offset[i], 1,
		    mpi_halo_t[i / 2], neighbor_rank[i], TAG);
    grid_comm.Recv(tile.params + tile.halo_recv_offset[i], 1,
		    mpi_halo_t[i / 2], neighbor_rank[i], TAG);
  }
  memcpy(tile.data_new, tile.data_old, tile.data_size[DIM_X] *
		  tile.data_size[DIM_Y] * sizeof(float));


  /****************************************************************************/
  std::size_t x_begin = 1 + (neighbor_rank[LEFT] == MPI::PROC_NULL);
  std::size_t x_end = tile.data_size[DIM_X] - (1 + (neighbor_rank[RIGHT] == MPI::PROC_NULL));
  std::size_t y_begin = 1 + (neighbor_rank[UPPER] == MPI::PROC_NULL);
  std::size_t y_end = tile.data_size[DIM_Y] - (1 + (neighbor_rank[LOWER] == MPI::PROC_NULL));

  for (std::size_t iter = 0; iter < parameters.nIterations; ++iter) {
    if (neighbor_rank[UPPER] != MPI::PROC_NULL) { //do I have upper neighbor?
      for (std::size_t col = x_begin; col < x_end; ++col) {
        ComputePoint(tile.data_old, tile.data_new, tile.params, tile.map,
			1, col, tile.data_size[DIM_X], parameters.airFlowRate,
			materialProperties.coolerTemp);
      }
      grid_comm.Isend(tile.data_new + tile.halo_send_offset[UPPER], 1,
		      mpi_halo_t[DIM_Y], neighbor_rank[UPPER], TAG);
    }
    if (neighbor_rank[LOWER] != MPI::PROC_NULL) { //do I have lower neighbor?
      for (std::size_t col = x_begin; col < x_end; ++col) {
        ComputePoint(tile.data_old, tile.data_new, tile.params, tile.map,
          		tile.data_size[DIM_Y] - 2, col, tile.data_size[DIM_X],
          		parameters.airFlowRate, materialProperties.coolerTemp);
      }
      grid_comm.Isend(tile.data_new + tile.halo_send_offset[LOWER], 1,
		      mpi_halo_t[DIM_Y], neighbor_rank[LOWER], TAG);
    }
    if (neighbor_rank[LEFT] != MPI::PROC_NULL) { //do I have left neighbor?
      for (std::size_t row = y_begin; row < y_end; ++row) {
        ComputePoint(tile.data_old, tile.data_new, tile.params, tile.map,
          		row, 1, tile.data_size[DIM_X], parameters.airFlowRate,
          		materialProperties.coolerTemp);
      }
      grid_comm.Isend(tile.data_new + tile.halo_send_offset[LEFT], 1,
		      mpi_halo_t[DIM_X], neighbor_rank[LEFT], TAG);
    }
    if (neighbor_rank[RIGHT] != MPI::PROC_NULL) { //do I have right neighbor?
      for (std::size_t row = y_begin; row < y_end; ++row) {
        ComputePoint(tile.data_old, tile.data_new, tile.params, tile.map,
          		row, tile.data_size[DIM_X] - 2, tile.data_size[DIM_X],
          		parameters.airFlowRate, materialProperties.coolerTemp);
      }
      grid_comm.Isend(tile.data_new + tile.halo_send_offset[RIGHT], 1,
		      mpi_halo_t[DIM_X], neighbor_rank[RIGHT], TAG);
    }

    /* Compute all other (inside) points. */
    for (std::size_t row = 2; row < tile.data_size[DIM_Y] - 2; ++row) {
      for (std::size_t col = 2; col < tile.data_size[DIM_X] - 2; ++col) {
        ComputePoint(tile.data_old, tile.data_new, tile.params, tile.map,
			row, col, tile.data_size[DIM_X], parameters.airFlowRate,
			materialProperties.coolerTemp);
      }
    }

    //grid_comm.Recv(tile.data_new + tile.halo_recv_offset[LOWER], 1, mpi_halo_t[DIM_Y], neighbor_rank[LOWER], TAG);
    //grid_comm.Recv(tile.data_new + tile.halo_recv_offset[UPPER], 1, mpi_halo_t[DIM_Y], neighbor_rank[UPPER], TAG);
    //grid_comm.Recv(tile.data_new + tile.halo_recv_offset[RIGHT], 1, mpi_halo_t[DIM_X], neighbor_rank[RIGHT], TAG);
    //grid_comm.Recv(tile.data_new + tile.halo_recv_offset[LEFT], 1, mpi_halo_t[DIM_X], neighbor_rank[LEFT], TAG);
    /* Receive halo zones. */
    for (std::size_t i = 0; i < NDIMS * NDIMS; ++i) {
      grid_comm.Recv(tile.data_old + tile.halo_recv_offset[i], 1, mpi_halo_t[i / 2], neighbor_rank[i], TAG);
    }

    swap(tile.data_old, tile.data_new);
  }

  //for (int i = 0; i < grid_size; ++i) {
  //  if (grid_rank == i) {
  //    printf("rank %d:\n", grid_rank);
  //    for (std::size_t i = 0; i < tile.data_size[DIM_Y]; ++i) {
  //      for (std::size_t j = 0; j < tile.data_size[DIM_X]; ++j) {
  //        printf("%3.0f ", tile.data_old[i * tile.data_size[DIM_X] + j]);
  //      }
  //      putchar('\n');
  //    }
  //    putchar('\n');
  //    fflush(stdout);
  //  }
  //  MPI::COMM_WORLD.Barrier();
  //}
  /****************************************************************************/


  /* Gather tiles into board. All processors. */
  grid_comm.Gatherv(tile.data_old, 1, mpi_tile_t, parResult, sendcounts,
		  displs, mpi_board_t, ROOT_PROC);
  if (world_rank == ROOT_PROC) {
    for (int i = 0; i < materialProperties.edgeSize; ++i) {
      for (int j = 0; j < materialProperties.edgeSize; ++j) {
        printf("%3.0f ", parResult[i * materialProperties.edgeSize + j]);
      }
      putchar('\n');
    }
  }


  /* Free memory. All processors. */
  _mm_free(tile.params);
  _mm_free(tile.map);
  _mm_free(tile.data_new);
  _mm_free(tile.data_old);

  /* Close the output file. Root processor only. */
  if (world_rank == ROOT_PROC && file_id != H5I_INVALID_HID) {
    H5Fclose(file_id);
  }

} // end of ParallelHeatDistribution
//------------------------------------------------------------------------------


/**
 * Store time step into output file (as a new dataset in Pixie format
 * @param [in] h5fileID   - handle to the output file
 * @param [in] data       - data to write
 * @param [in] edgeSize   - size of the domain
 * @param [in] snapshotId - snapshot id
 * @param [in] iteration  - id of iteration);
 */
void StoreDataIntoFile(hid_t         h5fileId,
                       const float  *data,
                       const size_t  edgeSize,
                       const size_t  snapshotId,
                       const size_t  iteration)
{
  hid_t   dataset_id, dataspace_id, group_id, attribute_id;
  hsize_t dims[2] = {edgeSize, edgeSize};

  string groupName = "Timestep_" + to_string((unsigned long long) snapshotId);

  // Create a group named "/Timestep_snapshotId" in the file.
  group_id = H5Gcreate(h5fileId,
                       groupName.c_str(),
                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);


  // Create the data space. (2D matrix)
  dataspace_id = H5Screate_simple(2, dims, NULL);

  // create a dataset for temperature and write data
  string datasetName = "Temperature";
  dataset_id = H5Dcreate(group_id,
                         datasetName.c_str(),
                         H5T_NATIVE_FLOAT,
                         dataspace_id,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset_id,
           H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           data);

  // close dataset
  H5Sclose(dataspace_id);


  // write attribute
  string atributeName="Time";
  dataspace_id = H5Screate(H5S_SCALAR);
  attribute_id = H5Acreate2(group_id, atributeName.c_str(),
                            H5T_IEEE_F64LE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT);

  double snapshotTime = double(iteration);
  H5Awrite(attribute_id, H5T_IEEE_F64LE, &snapshotTime);
  H5Aclose(attribute_id);


  // Close the dataspace.
  H5Sclose(dataspace_id);

  // Close to the dataset.
  H5Dclose(dataset_id);
} // end of StoreDataIntoFile
//------------------------------------------------------------------------------


/**
 * Main function of the project
 * @param [in] argc
 * @param [in] argv
 * @return
 */
int main(int argc, char *argv[])
{
  int rank, size;

  ParseCommandline(argc, argv, parameters);

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Get MPI rank and size
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  if (rank == 0)
  {
    // Create material properties and load from file
    materialProperties.LoadMaterialData(parameters.materialFileName, true);
    parameters.edgeSize = materialProperties.edgeSize;

    parameters.PrintParameters();
  }
  else
  {
    // Create material properties and load from file
    materialProperties.LoadMaterialData(parameters.materialFileName, false);
    parameters.edgeSize = materialProperties.edgeSize;
  }

  if (parameters.edgeSize % size)
  {
    if (rank == 0)
      printf("ERROR: number of MPI processes is not a divisor of N\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (parameters.IsRunSequntial())
  {
    if (rank == 0)
    {
      // Memory allocation for output matrices.
      seqResult = (float*)_mm_malloc(materialProperties.nGridPoints * sizeof(float), DATA_ALIGNMENT);

      SequentialHeatDistribution(seqResult,
                                 materialProperties,
                                 parameters,
                                 parameters.outputFileName);
    }
  }

  if (parameters.IsRunParallel())
  {
    // Memory allocation for output matrix.
    if (rank == 0)
      parResult = (float*) _mm_malloc(materialProperties.nGridPoints * sizeof(float), DATA_ALIGNMENT);
    else
      parResult = NULL;

    ParallelHeatDistribution(parResult,
                             materialProperties,
                             parameters,
                             parameters.outputFileName);
  }

  // Validate the outputs
  if (parameters.IsValidation() && rank == 0)
  {
    if (parameters.debugFlag)
    {
      printf("---------------- Sequential results ---------------\n");
      PrintArray(seqResult, materialProperties.edgeSize);

      printf("----------------- Parallel results ----------------\n");
      PrintArray(parResult, materialProperties.edgeSize);
    }

    if (VerifyResults(seqResult, parResult, parameters, 0.001f))
      printf("Verification OK\n");
    else
      printf("Verification FAILED\n");
  }

  /* Memory deallocation*/
  _mm_free(seqResult);
  _mm_free(parResult);

  MPI_Finalize();

  return EXIT_SUCCESS;
} // end of main
//------------------------------------------------------------------------------
