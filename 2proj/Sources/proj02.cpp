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
  float *newTemp = tempArray;
  float *oldTemp = seqResult;

  if (!parameters.batchMode) 
    printf("Starting sequential simulation... \n");
  
  //---------------------- [5] start the stop watch ------------------------------//
  double elapsedTime = MPI_Wtime();
  size_t i, j;
  size_t iteration;
  float middleColAvgTemp = 0.0f;

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
    if ((iteration % (parameters.nIterations / 10l)) == 
        ((parameters.nIterations / 10l) - 1l) && !parameters.batchMode)
    {
      //printf("Progress %ld%% (Average Temperature %.2f degrees)\n", 
      //       iteration / (parameters.nIterations / 100) + 1, 
      //       middleColAvgTemp);
      printf("it %lu/%lu: avg temp = %f\n", iteration, parameters.nIterations,
		      middleColAvgTemp);
    }
  } // for iteration

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

//--------------------------------------------------------------------------//
//---------------- THE SECTION WHERE STUDENTS MAY ADD CODE -----------------//
//--------------------------------------------------------------------------//
#define ROOT_PROC 0
#define TAG 0

#define DIM_X 0
#define DIM_Y 1
#define NDIMS 2

#define LEFT   0
#define RIGHT  1
#define UPPER  2
#define LOWER  3
#define NHALOS 4

struct grid_struct {
  MPI::Cartcomm comm;
  int comm_rank, comm_size;
};

struct middle_col_struct {
  MPI::Intracomm comm;
  int comm_rank, comm_size, root_grid_rank;
  double avg_temp;
};

struct tile_struct {
  std::size_t size[NDIMS]; //one tile size
  std::size_t data_size[NDIMS]; //actual data dimensions

  float *data_old, *data_new; //one tile t and t + 1 data
  int *map; //material map
  float *params; //material properties

  std::size_t halo_send_offset[NHALOS]; //halo zones offsets
  std::size_t halo_recv_offset[NHALOS]; //halo zones offsets
  std::size_t data_beg[NDIMS], data_end[NDIMS]; //data boundaries

  int coords[NDIMS]; //cartesian tile coordinates
  int neigh_rank[NDIMS * NDIMS]; //tile neighborhood ranks
};


void fill_grid_struct(grid_struct &grid, const MPI::Intracomm old_comm,
		const std::size_t tiles[NDIMS])
{
  const int dims[NDIMS] = {static_cast<const int>(tiles[DIM_X]),
	  static_cast<const int>(tiles[DIM_Y])};
  const bool periods[NDIMS] = {false, false};

  grid.comm = old_comm.Create_cart(NDIMS, dims, periods, true);
  grid.comm_rank = grid.comm.Get_rank();
  grid.comm_size = grid.comm.Get_size();
}

void fill_tile_struct(tile_struct &tile, const grid_struct grid,
		const int edge_size, const std::size_t tiles[NDIMS])
{
  /* Memory and actual data sizes. */
  tile.size[DIM_X] = edge_size / tiles[DIM_X];
  tile.size[DIM_Y] = edge_size / tiles[DIM_Y];
  tile.data_size[DIM_X] = tile.size[DIM_X] + 2; //for halo zones
  tile.data_size[DIM_Y] = tile.size[DIM_Y] + 2; //for halo zones

  /* Halo zones position offsets in memory. */
  tile.halo_send_offset[LEFT] = tile.data_size[DIM_X] + 1;
  tile.halo_send_offset[RIGHT] = tile.data_size[DIM_X] * 2 - 2;
  tile.halo_send_offset[UPPER] = tile.data_size[DIM_X] + 1;
  tile.halo_send_offset[LOWER] = tile.data_size[DIM_X] * (tile.data_size[DIM_Y] - 2) + 1;
  tile.halo_recv_offset[LEFT] = tile.data_size[DIM_X];
  tile.halo_recv_offset[RIGHT] = tile.data_size[DIM_X] * 2 - 1;
  tile.halo_recv_offset[UPPER] = 1;
  tile.halo_recv_offset[LOWER] = tile.data_size[DIM_X] * (tile.data_size[DIM_Y] - 1) + 1;

  /* Memory allocation. */
  tile.data_old = static_cast<float *>(_mm_malloc(tile.data_size[DIM_X] *
			  tile.data_size[DIM_Y] * sizeof(float), DATA_ALIGNMENT));
  tile.data_new = static_cast<float *>(_mm_malloc(tile.data_size[DIM_X] *
			  tile.data_size[DIM_Y] * sizeof(float), DATA_ALIGNMENT));
  tile.map = static_cast<int *>(_mm_malloc(tile.data_size[DIM_X] *
			  tile.data_size[DIM_Y] * sizeof(int), DATA_ALIGNMENT));
  tile.params = static_cast<float *>(_mm_malloc(tile.data_size[DIM_X] *
			  tile.data_size[DIM_Y] * sizeof(float), DATA_ALIGNMENT));
  if (!(tile.data_old && tile.data_new && tile.map && tile.params)) {
    MPI::COMM_WORLD.Abort(1);
  }

  /* Store coordinates of each tile. */
  grid.comm.Get_coords(grid.comm.Get_rank(), NDIMS, tile.coords);

  /* Calculate and store neighborhood ranks in grid topology. */
  for (std::size_t dim = 0, index = 0; dim < NDIMS; dim++) { //X, Y
    int dummy;

    grid.comm.Shift(dim, -1, dummy, tile.neigh_rank[index++]); //left/up
    grid.comm.Shift(dim, 1, dummy, tile.neigh_rank[index++]); //right/down
  }

  /* Loop indexes have to differ according to neighborhood - if the processor
   * is on the edge of the board, there is additional border, which shouldn't be
   * altered by ComputePoint.
   */
  tile.data_beg[DIM_X] = 1 + (tile.neigh_rank[LEFT] == MPI::PROC_NULL);
  tile.data_end[DIM_X] = tile.data_size[DIM_X] - (1 + (tile.neigh_rank[RIGHT] == MPI::PROC_NULL));
  tile.data_beg[DIM_Y] = 1 + (tile.neigh_rank[UPPER] == MPI::PROC_NULL);
  tile.data_end[DIM_Y] = tile.data_size[DIM_Y] - (1 + (tile.neigh_rank[LOWER] == MPI::PROC_NULL));
}

void fill_middle_col_struct(middle_col_struct &middle_col,
		const grid_struct grid, const std::size_t tiles[NDIMS])
{
  std::size_t index = 0;
  int middle_col_ranks[tiles[DIM_Y]], grid_root = ROOT_PROC;

  for (int rank = 0; rank < grid.comm_size; ++rank) {
    int rank_coords[NDIMS];

    grid.comm.Get_coords(rank, NDIMS, rank_coords);
    if (rank_coords[DIM_X] == (tiles[DIM_X] / 2)) {
      middle_col_ranks[index++] = rank;
    }
  }

  MPI::Group grid_group = grid.comm.Get_group();
  MPI::Group middle_col_group = grid_group.Incl(tiles[DIM_Y], middle_col_ranks);

  middle_col_group.Translate_ranks(middle_col_group, 1, &grid_root, grid_group,
		  &middle_col.root_grid_rank);
  middle_col.comm = grid.comm.Create(middle_col_group);

  if (middle_col.comm != MPI::COMM_NULL) {
    middle_col.comm_size = middle_col.comm.Get_size();
    middle_col.comm_rank = middle_col.comm.Get_rank();
  }
  middle_col.avg_temp = 0.0;
}

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
  struct {
    MPI::Intracomm comm = MPI::COMM_WORLD;
    const int comm_rank = MPI::COMM_WORLD.Get_rank();
    const int comm_size = MPI::COMM_WORLD.Get_size();

    std::size_t tiles[NDIMS]; //tiles per one dimension
  } board;

  tile_struct tile;
  grid_struct grid;
  middle_col_struct middle_col;

  /* Create a new output hdf5 file. Root processor only. */
  hid_t file_id = H5I_INVALID_HID;
  if (board.comm_rank == ROOT_PROC) {
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

  /* Partition input board into smaller parts - tiles. */
  board.tiles[DIM_X] = std::sqrt(board.comm_size / (1 + (static_cast<std::size_t>(
					  std::log2(board.comm_size))& 1)));
  board.tiles[DIM_Y] = board.comm_size / board.tiles[DIM_X];

  /* Create a new communicator - 2D grid Cartesian topology. */
  fill_grid_struct(grid, board.comm, board.tiles);

  /* Fill the tile struct - dimensions, halo zones offsets, memory. */
  fill_tile_struct(tile, grid, materialProperties.edgeSize, board.tiles);

  /* Create a new communicator for middle row. */
  fill_middle_col_struct(middle_col, grid, board.tiles);

  /* Create a new MPI data type - subarray corresponding to one tile
   * among whole data board.
   */
  int sizes[NDIMS] = {materialProperties.edgeSize, materialProperties.edgeSize};
  int subsizes[NDIMS] = {tile.size[DIM_Y], tile.size[DIM_X]}; //rows, then cols
  int starts[NDIMS] = {0, 0};
  MPI::Datatype mpi_board_t = MPI::FLOAT.Create_subarray(NDIMS, sizes, subsizes,
		  starts, MPI::ORDER_C);
  mpi_board_t = mpi_board_t.Create_resized(0, tile.size[DIM_X] * sizeof(float));
  mpi_board_t.Commit();

  /* Create a new MPI data type - subarray corresponding to one tile
   * in array with data and halo zones.
   */
  sizes[DIM_X] = tile.data_size[DIM_Y]; //rows first
  sizes[DIM_Y] = tile.data_size[DIM_X]; //cols second
  starts[DIM_X] = starts[DIM_Y] = 1; //leave halo zones untouched
  MPI::Datatype mpi_tile_data_t = MPI::FLOAT.Create_subarray(NDIMS, sizes, subsizes,
		  starts, MPI::ORDER_C);
  MPI::Datatype mpi_tile_map_t = MPI::INT.Create_subarray(NDIMS, sizes, subsizes,
		  starts, MPI::ORDER_C);
  mpi_tile_data_t.Commit();
  mpi_tile_map_t.Commit();

  /* Calculate displacement in data board for each tile according to processor
   * coordinates. Root processor only.
   */
  int sendcounts[grid.comm_size], displs[grid.comm_size];
  if (grid.comm_rank == ROOT_PROC) {
    for (int rank = 0; rank < grid.comm_size; ++rank) {
      int rank_coords[NDIMS];

      grid.comm.Get_coords(rank, NDIMS, rank_coords);
      sendcounts[rank] = 1;
      displs[rank] = rank_coords[0] + rank_coords[1] * board.tiles[DIM_X] *
	      tile.size[DIM_Y];
    }
  }

  /* Scatter initial temperature, domain map and parameters into tiles. */
  grid.comm.Scatterv(materialProperties.initTemp, sendcounts, displs,
		  mpi_board_t, tile.data_old, 1, mpi_tile_data_t, ROOT_PROC);
  grid.comm.Scatterv(materialProperties.domainMap, sendcounts, displs,
		  mpi_board_t, tile.map, 1, mpi_tile_map_t, ROOT_PROC);
  grid.comm.Scatterv(materialProperties.domainParams, sendcounts, displs,
		  mpi_board_t, tile.params, 1, mpi_tile_data_t, ROOT_PROC);

  /* Create row and column halo data types. */
  MPI::Datatype mpi_halo_t[NDIMS];
  mpi_halo_t[DIM_X] = MPI::FLOAT.Create_vector(tile.size[DIM_Y], 1,
		  tile.data_size[DIM_X]); //column halo
  mpi_halo_t[DIM_Y] = MPI::FLOAT.Create_vector(1, tile.size[DIM_X],
		  tile.data_size[DIM_X]); //row halo
  mpi_halo_t[DIM_X].Commit();
  mpi_halo_t[DIM_Y].Commit();

  /* Initialization - send and reveive halo zones for data nad parameters. */
  for (std::size_t i = 0; i < NDIMS * NDIMS; ++i) {
    grid.comm.Isend(tile.data_old + tile.halo_send_offset[i], 1,
		    mpi_halo_t[i / 2], tile.neigh_rank[i], TAG);
    grid.comm.Isend(tile.params + tile.halo_send_offset[i], 1,
		    mpi_halo_t[i / 2], tile.neigh_rank[i], TAG);
  }
  for (std::size_t i = 0; i < NDIMS * NDIMS; ++i) {
    grid.comm.Recv(tile.data_old + tile.halo_recv_offset[i], 1,
		    mpi_halo_t[i / 2], tile.neigh_rank[i], TAG);
    grid.comm.Recv(tile.params + tile.halo_recv_offset[i], 1,
		    mpi_halo_t[i / 2], tile.neigh_rank[i], TAG);
  }
  memcpy(tile.data_new, tile.data_old, tile.data_size[DIM_X] *
		  tile.data_size[DIM_Y] * sizeof(float));

  /* Start measuring time. */
  board.comm.Barrier();
  double elapsed_time = MPI::Wtime();

  /* For all iterations. */
  for (std::size_t iter = 0; iter < parameters.nIterations; ++iter) {
    /* Conditionally compute and send halo zones. */
    if (tile.neigh_rank[LEFT] != MPI::PROC_NULL) { //do I have left neighbor?
      for (std::size_t row = tile.data_beg[DIM_Y]; row < tile.data_end[DIM_Y]; ++row) {
        ComputePoint(tile.data_old, tile.data_new, tile.params, tile.map,
          		row, 1, tile.data_size[DIM_X], parameters.airFlowRate,
          		materialProperties.coolerTemp);
      }
      grid.comm.Isend(tile.data_new + tile.halo_send_offset[LEFT], 1,
		      mpi_halo_t[DIM_X], tile.neigh_rank[LEFT], TAG);
    }
    if (tile.neigh_rank[RIGHT] != MPI::PROC_NULL) { //do I have right neighbor?
      for (std::size_t row = tile.data_beg[DIM_Y]; row < tile.data_end[DIM_Y]; ++row) {
        ComputePoint(tile.data_old, tile.data_new, tile.params, tile.map,
          		row, tile.data_size[DIM_X] - 2, tile.data_size[DIM_X],
          		parameters.airFlowRate, materialProperties.coolerTemp);
      }
      grid.comm.Isend(tile.data_new + tile.halo_send_offset[RIGHT], 1,
		      mpi_halo_t[DIM_X], tile.neigh_rank[RIGHT], TAG);
    }
    if (tile.neigh_rank[UPPER] != MPI::PROC_NULL) { //do I have upper neighbor?
      for (std::size_t col = tile.data_beg[DIM_X]; col < tile.data_end[DIM_X]; ++col) {
        ComputePoint(tile.data_old, tile.data_new, tile.params, tile.map,
			1, col, tile.data_size[DIM_X], parameters.airFlowRate,
			materialProperties.coolerTemp);
      }
      grid.comm.Isend(tile.data_new + tile.halo_send_offset[UPPER], 1,
		      mpi_halo_t[DIM_Y], tile.neigh_rank[UPPER], TAG);
    }
    if (tile.neigh_rank[LOWER] != MPI::PROC_NULL) { //do I have lower neighbor?
      for (std::size_t col = tile.data_beg[DIM_X]; col < tile.data_end[DIM_X]; ++col) {
        ComputePoint(tile.data_old, tile.data_new, tile.params, tile.map,
          		tile.data_size[DIM_Y] - 2, col, tile.data_size[DIM_X],
          		parameters.airFlowRate, materialProperties.coolerTemp);
      }
      grid.comm.Isend(tile.data_new + tile.halo_send_offset[LOWER], 1,
		      mpi_halo_t[DIM_Y], tile.neigh_rank[LOWER], TAG);
    }

    /* Compute all other (inside) points. */
    for (std::size_t row = 2; row < tile.data_size[DIM_Y] - 2; ++row) {
      for (std::size_t col = 2; col < tile.data_size[DIM_X] - 2; ++col) {
        ComputePoint(tile.data_old, tile.data_new, tile.params, tile.map,
			row, col, tile.data_size[DIM_X], parameters.airFlowRate,
			materialProperties.coolerTemp);
      }
    }

    /* Compute the average temperature in the middle column. */
    if (middle_col.comm != MPI::COMM_NULL) {
      double local_avg = 0.0;
      const std::size_t middle_col_pos = (board.tiles[DIM_X] < 2) ?
              tile.data_size[DIM_X] / 2 : 1;

      for (std::size_t row = 1; row < tile.data_size[DIM_Y] - 1; ++row) {
        local_avg += tile.data_new[row * tile.data_size[DIM_X] +
      	  middle_col_pos];
      }
      local_avg /= tile.size[DIM_Y];

      middle_col.comm.Reduce(&local_avg, &middle_col.avg_temp, 1, MPI::DOUBLE,
		      MPI::SUM, ROOT_PROC);
      middle_col.avg_temp /= middle_col.comm_size;

      if ((middle_col.comm_rank == ROOT_PROC) && 
          	    (iter % (parameters.nIterations / 10l)) ==
          	    ((parameters.nIterations / 10l) - 1l) &&
          	    !parameters.batchMode) {
        printf("it %lu/%lu: avg temp = %f\n", iter, parameters.nIterations,
			middle_col.avg_temp);
      }

      if ((middle_col.comm_rank == ROOT_PROC) &&
		      (iter == parameters.nIterations - 1)) {
        grid.comm.Isend(&middle_col.avg_temp, 1, MPI::DOUBLE, ROOT_PROC, TAG);
      }
    }

    /* Store time step in the output file if necessary. */
    if ((iter % parameters.diskWriteIntensity) == 0) {
      /* Gather tiles into board. */
      grid.comm.Gatherv(tile.data_new, 1, mpi_tile_data_t, parResult,
		      sendcounts, displs, mpi_board_t, ROOT_PROC);
      
      if (file_id != H5I_INVALID_HID && grid.comm_rank == ROOT_PROC) {
        StoreDataIntoFile(file_id, parResult, materialProperties.edgeSize,
			iter / parameters.diskWriteIntensity, iter);
      }
    }

    /* Receive halo zones. */
    for (std::size_t i = 0; i < NDIMS * NDIMS; ++i) {
      grid.comm.Recv(tile.data_new + tile.halo_recv_offset[i], 1, mpi_halo_t[i / 2], tile.neigh_rank[i], TAG);
    }

    std::swap(tile.data_old, tile.data_new);
  }

  /* Finish measuring time. */
  board.comm.Barrier();
  elapsed_time = MPI::Wtime() - elapsed_time;

  /* Print final result. */
  if (board.comm_rank == ROOT_PROC) {
    if (parameters.nIterations > 0) {
      grid.comm.Recv(&middle_col.avg_temp, 1, MPI::DOUBLE, middle_col.root_grid_rank, TAG);
    }
    if (!parameters.batchMode) {
      printf("\nExecution time of parallel version %.5f\n", elapsed_time);
    } else {
      printf("%s;%s;%f;%e;%e\n", outputFileName.c_str(), "par", middle_col.avg_temp,
          	    elapsed_time, elapsed_time / parameters.nIterations);
    }
  }

  /* Gather tiles into board. */
  grid.comm.Gatherv(tile.data_old, 1, mpi_tile_data_t, parResult, sendcounts,
		  displs, mpi_board_t, ROOT_PROC);

  /* Free memory. */
  _mm_free(tile.params);
  _mm_free(tile.map);
  _mm_free(tile.data_new);
  _mm_free(tile.data_old);

  /* Close the output file. Root processor only. */
  if (board.comm_rank == ROOT_PROC && file_id != H5I_INVALID_HID) {
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
