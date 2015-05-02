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
      printf("Progress %ld%% (Average Temperature %.2f degrees)\n", 
             iteration / (parameters.nIterations / 100) + 1, 
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
