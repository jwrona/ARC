/**
 * @file        proj01.cpp
 * @author      Jiri Jaros and Vojtech Nikl\n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       Parallelisation of Heat Distribution Method in Heterogenous
 *              Media using OpenMP
 *
 * @version     2015
 * @date        19 February 2015, 16:22 (created) \n
 *              11 March    2015, 23:12 (last revised)
 *
 * @detail
 * This is the main file of the project. Add all code here.
 */


#include <string.h>
#include <string>

#include <omp.h>
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
float * seqResult = NULL;
/// Temperature data for parallel method.
float * parResult = NULL;

/// Parameters of the simulation
TParameters parameters;


//----------------------------------------------------------------------------//
//------------------------- Function declarations ----------------------------//
//----------------------------------------------------------------------------//

/// Sequential implementation of the Heat distribution
void SequentialHeatDistribution(float *                     seqResult,
                                const TMaterialProperties & materialProperties,
                                const TParameters         & parameters,
                                string                      outputFileName);

/// Parallel Implementation of the Heat distribution (Non-overlapped file output)
void ParallelHeatDistributionNonOverlapped(float *                     parResult,
                                           const TMaterialProperties & materialProperties,
                                           const TParameters         & parameters,
                                           string                      outputFileName);
                                           
/// Parallel Implementation of the Heat distribution (Overlapped file output)
void ParallelHeatDistributionOverlapped(float *                     parResult,
                                        const TMaterialProperties & materialProperties,
                                        const TParameters         & parameters,
                                        string                      outputFileName);                                           

/// Store time step into output file
void StoreDataIntoFile(hid_t         h5fileId,
                       const float * data,
                       const size_t  edgeSize,
                       const size_t  snapshotId,
                       const size_t  iteration);


//----------------------------------------------------------------------------//
//------------------------- Function implementation  -------------------------//
//----------------------------------------------------------------------------//


/**
 * Sequential version of the Heat distribution in heterogenous 2D medium
 * @param [out] seqResult          - Final heat distribution
 * @param [in]  materialProperties - Material properties
 * @param [in]  parameters         - parameters of the simulation
 * @param [in]  outputFileName     - Output file name (if NULL string, do not store)
 *
 */
void SequentialHeatDistribution(float *                     seqResult,
                                const TMaterialProperties & materialProperties,
                                const TParameters         & parameters,
                                string                      outputFileName)
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
  float * tempArray = (float *) _mm_malloc(materialProperties.nGridPoints * 
                                           sizeof(float), DATA_ALIGNMENT);
                                           
  // [3] init of arrays
  for  (size_t i = 0; i < materialProperties.nGridPoints; i++)
  {
    tempArray[i] = materialProperties.initTemp[i];
    seqResult[i] = materialProperties.initTemp[i];
  }

  // [4] t+1 values
  float * newTemp = seqResult;
  // t - values
  float * oldTemp = tempArray;

  if (!parameters.batchMode) 
    printf("Starting sequential simulation... \n");
  
  //---------------------- [5] press the stop watch ------------------------------//
  double elapsedTime = omp_get_wtime();
  size_t i, j;
  size_t iteration;
  float middleColAvgTemp = 0.0f;

  // [6] Start the iterative simulation
  for (iteration = 0; iteration < parameters.nIterations; iteration++)
  {
    // calculate one iteration of the heat distribution
    // We skip the grid points at the edges
    for (i = 1; i < materialProperties.edgeSize - 1; i++)
    {
      for (j = 1; j < materialProperties.edgeSize - 1; j++)
      {
        // [a)] Calculate neighbor indices
        const int center =  i * materialProperties.edgeSize + j;
        const int top    =  center - materialProperties.edgeSize;
        const int bottom =  center + materialProperties.edgeSize;
        const int left   =  center - 1;
        const int right  =  center + 1;

        // [b)] The reciprocal value of the sum of domain parameters for normalization
        const float frec = 1.0f / (materialProperties.domainParams[top]    +
                                   materialProperties.domainParams[bottom] +
                                   materialProperties.domainParams[left]   +
                                   materialProperties.domainParams[center] +
                                   materialProperties.domainParams[right]);

        // [c)] Calculate new temperature in the grid point
        float pointTemp = 
              oldTemp[top]    * materialProperties.domainParams[top]    * frec +
              oldTemp[bottom] * materialProperties.domainParams[bottom] * frec +
	            oldTemp[left]   * materialProperties.domainParams[left]   * frec +
	            oldTemp[right]  * materialProperties.domainParams[right]  * frec +
    	        oldTemp[center] * materialProperties.domainParams[center] * frec;
	        

        // [d)] Remove some of the heat due to air flow (5% of the new air)
        pointTemp = (materialProperties.domainMap[center] == 0)  
                    ? (parameters.airFlowRate * materialProperties.CoolerTemp) + ((1.f - parameters.airFlowRate) * pointTemp)
                    : pointTemp;

        newTemp[center] = pointTemp;

      } // for j
    }// for i

    // [7] Calculate average temperature in the middle column
    middleColAvgTemp = 0.0f;
    
    for (i = 0; i < materialProperties.edgeSize; i++)
    {
      middleColAvgTemp += newTemp[i*materialProperties.edgeSize +
                          materialProperties.edgeSize/2];
    }
   
    middleColAvgTemp /= materialProperties.edgeSize;                         

    // [8] Store time step in the output file if necessary
    if ((file_id != H5I_INVALID_HID)  && ((iteration % parameters.diskWriteIntensity) == 0))
    {
      StoreDataIntoFile(file_id,
                        newTemp,
                        materialProperties.edgeSize,
                        iteration / parameters.diskWriteIntensity,
                        iteration);
    }

    // [9] Swap new and old values
    swap(newTemp, oldTemp);

    // [10] Print progress and average temperature of the middle column
    if ((iteration % (parameters.nIterations / 10l)) == 
        ((parameters.nIterations / 10l) - 1l) && !parameters.batchMode)
    {
      printf("Progress %ld%% (Average Temperature %.2f degrees)\n", 
             iteration / ( parameters.nIterations / 100) + 1, 
             middleColAvgTemp);
    }

  }// for iteration

  //-------------------- stop the stop watch  --------------------------------//  
  double totalTime = omp_get_wtime() - elapsedTime;
  
  // [11] Print final result
  if (!parameters.batchMode)
    printf("\nExecution time of sequential version %.5f\n", totalTime);
  else
    printf("%s;%s;%f;%e;%e\n", outputFileName.c_str(), "seq", 
                               middleColAvgTemp, totalTime, 
                               totalTime / parameters.nIterations);   

  // Close the output file
  if (file_id != H5I_INVALID_HID) H5Fclose(file_id);

  // [12] Return correct results in the correct array
  if (iteration & 1)
  {
    memcpy(seqResult, tempArray, materialProperties.nGridPoints * sizeof(float));
  }

  _mm_free(tempArray);
}// end of SequentialHeatDistribution
//------------------------------------------------------------------------------


/**
 * Parallel version of the Heat distribution in heterogenous 2D medium
 * with non-overlapped file output. 
 * @param [out] parResult          - Final heat distribution
 * @param [in]  materialProperties - Material properties
 * @param [in]  parameters         - parameters of the simulation
 * @param [in]  outputFileName     - Output file name (if NULL string, do not store)
 *
 * @note This is one of two functions that students are allowed to change and 
 *       optimize. Do not modify the given code, add only openmp pragmas or 
 *       openmp calls.                                                      
 */
void ParallelHeatDistributionNonOverlapped(float *                     parResult,
                                           const TMaterialProperties & materialProperties,
                                           const TParameters         & parameters,
                                           string                      outputFileName)
{
  // Create a new output hdf5 file
  hid_t file_id = H5I_INVALID_HID;
  
  if (outputFileName != "")
  {
    if (outputFileName.find(".h5") == string::npos)
      outputFileName.append("_par1.h5");
    else
      outputFileName.insert(outputFileName.find_last_of("."), "_par1");
    
    file_id = H5Fcreate(outputFileName.c_str(),
                        H5F_ACC_TRUNC,
                        H5P_DEFAULT,
                        H5P_DEFAULT);
    if (file_id < 0) ios::failure("Cannot create output file");
  }

  // we need a temporary array to prevent mixing of data form step t and t+1
  float * tempArray = (float *) _mm_malloc(materialProperties.nGridPoints * sizeof(float),
                                           DATA_ALIGNMENT);
  // t+1 values
  float * newTemp = parResult;
  // t - values
  float * oldTemp = tempArray;

  if (!parameters.batchMode) 
    printf("\nStarting parallel simulation... \n");
  //---------------------- prest the stop watch ------------------------------//
  double elapsedTime = omp_get_wtime();
  size_t i, j;
  size_t iteration;
  float middleColAvgTemp = 0.0f;

  //--------------------------------------------------------------------------//
  //----- START OF THE SECTION WHERE STUDENTS MAY ADD/EDIT OMP PRAGMAS -------//
  //--------------------------------------------------------------------------//  
  #pragma omp parallel private(iteration, i, j)
  {
    #pragma omp for
    for  (i = 0; i < materialProperties.nGridPoints; i++)
    {
      tempArray[i] = materialProperties.initTemp[i];
      parResult[i] = materialProperties.initTemp[i];
    }
    
    for (iteration = 0; iteration < parameters.nIterations; iteration++)
    {
      middleColAvgTemp = 0.0f;
      // calculate one iteration of the heat distribution
      // We skip the grid points at the edges
      #pragma omp for
      for (i = 1; i < materialProperties.edgeSize - 1; i++)
      {
        for (j = 1; j < materialProperties.edgeSize - 1; j++)
        {
          // calculate neighbors
          const int center =  i * materialProperties.edgeSize + j;
          const int top    =  center - materialProperties.edgeSize;
          const int bottom =  center + materialProperties.edgeSize;
          const int left   =  center - 1;
          const int right  =  center + 1;

          // the reciprocal value of the sum of domain parameters for normalization
          const float frec = 1.0f / (materialProperties.domainParams[top]    +
                                     materialProperties.domainParams[bottom] +
                                     materialProperties.domainParams[left]   +
                                     materialProperties.domainParams[center] +
                                     materialProperties.domainParams[right]);

          // calculate new temperature in the grid point
          float pointTemp = 
              oldTemp[top]    * materialProperties.domainParams[top]    * frec +
              oldTemp[bottom] * materialProperties.domainParams[bottom] * frec +
              oldTemp[left]   * materialProperties.domainParams[left]   * frec +
              oldTemp[right]  * materialProperties.domainParams[right]  * frec +
    	        oldTemp[center] * materialProperties.domainParams[center] * frec;
	          

          // remove some of the heat due to air flow (5% of the new air)
          pointTemp = (materialProperties.domainMap[center] == 0)  
                      ? (parameters.airFlowRate * materialProperties.CoolerTemp) + 
                        ((1.f - parameters.airFlowRate) * pointTemp)
                      : pointTemp;

          newTemp[center] = pointTemp;

        } // for j
      }// for i

      //calculate average temperature in the middle column
      #pragma omp for reduction(+:middleColAvgTemp)
      for (i = 0; i < materialProperties.edgeSize; i++)
      {
        middleColAvgTemp += newTemp[i*materialProperties.edgeSize +
                            materialProperties.edgeSize/2];
      }
      
      #pragma omp master
      {
        middleColAvgTemp /= materialProperties.edgeSize;                         

        // Store time step in the output file if necessary
        if ((file_id != H5I_INVALID_HID)  && ((iteration % parameters.diskWriteIntensity) == 0))
        {
          StoreDataIntoFile(file_id,
                            newTemp,
                            materialProperties.edgeSize,
                            iteration / parameters.diskWriteIntensity,
                            iteration);
        }

        // swap new and old values
        swap(newTemp, oldTemp);
        
        if ((iteration % (parameters.nIterations / 10l)) == 
            ((parameters.nIterations / 10l) - 1l) && !parameters.batchMode)
        {
          printf("Progress %ld%% (Average Temperature %.2f degrees)\n", 
                 iteration / ( parameters.nIterations / 100) + 1, 
                 middleColAvgTemp);
        }
      }// pragma master
      #pragma omp barrier
    }// for iteration
  } // pragma parallel  

  //--------------------------------------------------------------------------//
  //--------- END OF THE SECTION WHERE STUDENTS MAY ADD OMP PRAGMAS ----------//
  //--------------------------------------------------------------------------//
  
  double totalTime = omp_get_wtime() - elapsedTime;
  
  if (!parameters.batchMode)
    printf("\nExecution time of parallel (mode 1) version %.5f\n", totalTime);
  else
    printf("%s;%s;%f;%e;%e\n", outputFileName.c_str(), "par1", 
                               middleColAvgTemp, totalTime, 
                               totalTime / parameters.nIterations);  

  //-------------------- stop the stop watch  --------------------------------//

  // Close the output file
  if (file_id != H5I_INVALID_HID) H5Fclose(file_id);

  // return correct results in the correct array
  if (parameters.nIterations & 1)
  {
    memcpy(parResult, tempArray, materialProperties.nGridPoints * sizeof(float));
  }

  _mm_free(tempArray);
}// end of ParallelHeatDistribution
//------------------------------------------------------------------------------
 



/**
 * Parallel version of the Heat distribution in heterogenous 2D medium
 * with overlapped file output. 
 * @param [out] parResult          - Final heat distribution
 * @param [in]  materialProperties - Material properties
 * @param [in]  parameters         - parameters of the simulation
 * @param [in]  outputFileName     - Output file name (if NULL string, do not store)
 *
 * @note This is one of two functions that students are allowed to change and 
 *       optimize. Do not modify the given code, add only openmp pragmas or 
 *       openmp calls.                                                      
 */
void ParallelHeatDistributionOverlapped(float *                     parResult,
                                        const TMaterialProperties & materialProperties,
                                        const TParameters         & parameters,
                                        string                      outputFileName)
{
  // Create a new output hdf5 file
  hid_t file_id = H5I_INVALID_HID;
  
  if (outputFileName != "")
  {
    if (outputFileName.find(".h5") == string::npos)
      outputFileName.append("_par2.h5");
    else
      outputFileName.insert(outputFileName.find_last_of("."), "_par2");
    
    file_id = H5Fcreate(outputFileName.c_str(),
                        H5F_ACC_TRUNC,
                        H5P_DEFAULT,
                        H5P_DEFAULT);
    if (file_id < 0) ios::failure("Cannot create output file");
  }

  // we need a temporary array to prevent mixing of data form step t and t+1
  float * tempArray = (float *) _mm_malloc(materialProperties.nGridPoints * sizeof(float),
                                           DATA_ALIGNMENT);
  // t+1 values
  float * newTemp = parResult;
  // t - values
  float * oldTemp = tempArray;

  if (!parameters.batchMode) 
    printf("\nStarting parallel simulation... \n");
    
  //---------------------- prest the stop watch ------------------------------//
  double elapsedTime = omp_get_wtime();
  size_t i, j;
  size_t iteration;
  float middleColAvgTemp = 0.0f;

  //--------------------------------------------------------------------------//
  //---------------------------- START OF YOUR CODE --------------------------//
  //--------------------------------------------------------------------------//   
  struct {
    float *data;
    size_t iteration;
    bool no_more_data;
    } buffer;

  buffer.data = (float*)_mm_malloc(materialProperties.nGridPoints * sizeof(float), DATA_ALIGNMENT);
  buffer.no_more_data = false;
  bool buffer_full = false;

  omp_set_nested(1);
  omp_set_num_threads(omp_get_max_threads() - 1);

  #pragma omp parallel sections num_threads(2)
  {
    /**************************************************************************/
    #pragma omp section /* Computation section. */
    {
      #pragma omp parallel private(iteration, i, j)
      {
        #pragma omp for /* Memory initialization. */
        for (i = 0; i < materialProperties.nGridPoints; ++i) {
          tempArray[i] = materialProperties.initTemp[i];
          parResult[i] = materialProperties.initTemp[i];
        }

        for (iteration = 0; iteration < parameters.nIterations; ++iteration) {
	  middleColAvgTemp = 0.0f;
          #pragma omp for /* Parallel loop through rows, serial through columns. */
          for (i = 1; i < materialProperties.edgeSize - 1; ++i) {
            for (j = 1; j < materialProperties.edgeSize - 1; ++j) {
              const int center =  i * materialProperties.edgeSize + j;
              const int top = center - materialProperties.edgeSize;
              const int bottom = center + materialProperties.edgeSize;
              const int left = center - 1;
              const int right = center + 1;

              // the reciprocal value of the sum of domain parameters for normalization
              const float frec = 1.0f / (materialProperties.domainParams[top]    +
                                         materialProperties.domainParams[bottom] +
                                         materialProperties.domainParams[left]   +
                                         materialProperties.domainParams[center] +
                                         materialProperties.domainParams[right]);

              // calculate new temperature in the grid point
              float pointTemp = 
                  oldTemp[top]    * materialProperties.domainParams[top]    * frec +
                  oldTemp[bottom] * materialProperties.domainParams[bottom] * frec +
                  oldTemp[left]   * materialProperties.domainParams[left]   * frec +
                  oldTemp[right]  * materialProperties.domainParams[right]  * frec +
        	        oldTemp[center] * materialProperties.domainParams[center] * frec;
                      

              // remove some of the heat due to air flow (5% of the new air)
              pointTemp = (materialProperties.domainMap[center] == 0)  
                          ? (parameters.airFlowRate * materialProperties.CoolerTemp) + 
                            ((1.f - parameters.airFlowRate) * pointTemp)
                          : pointTemp;

              newTemp[center] = pointTemp;

            }// for j
          }// for i

          #pragma omp for reduction(+:middleColAvgTemp) /* Sum of middle column cells temperature. */
          for (i = 0; i < materialProperties.edgeSize; i++) {
            middleColAvgTemp += newTemp[i * materialProperties.edgeSize + materialProperties.edgeSize / 2];
          }

          #pragma omp master
          {
            middleColAvgTemp /= materialProperties.edgeSize; /* From sum to average. */

            if (iteration % parameters.diskWriteIntensity == 0) {
              do { //wait until buffer is safe for writing
                #pragma omp flush(buffer_full)
              } while(buffer_full);

	      memcpy(buffer.data, newTemp, materialProperties.nGridPoints * sizeof(float));
              buffer.iteration = iteration;

              buffer_full = true; //buffer is safe for reading now
              #pragma omp flush(buffer_full, buffer)
            }

            if ((iteration % (parameters.nIterations / 10l)) == 
                ((parameters.nIterations / 10l) - 1l) && !parameters.batchMode)
            {
              printf("Progress %ld%% (Average Temperature %.2f degrees)\n", 
                     iteration / ( parameters.nIterations / 100) + 1, 
                     middleColAvgTemp);
            }

	    //swap new and old values
	    swap(newTemp, oldTemp);
          }
          #pragma omp barrier
        }// for iterations
      }// parallel

      do { //wait until buffer is safe for writing
        #pragma omp flush(buffer_full)
      } while(buffer_full);
      
      buffer.no_more_data = true;
      
      buffer_full = true; //buffer is safe for reading now
      #pragma omp flush(buffer_full, buffer)
    } /* End of omputation section. */
    /**************************************************************************/

    /**************************************************************************/
    #pragma omp section /* Disk write section. */
    {
      while (true) {
        do { //wait until buffer is safe for reading
          #pragma omp flush(buffer_full)
        } while(!buffer_full);
        #pragma omp flush(buffer)

        if (buffer.no_more_data) {
          break;
        }

        // Store time step in the output file if necessary
        if (file_id != H5I_INVALID_HID)
        {
          StoreDataIntoFile(file_id, buffer.data, materialProperties.edgeSize,
                            buffer.iteration / parameters.diskWriteIntensity,
                            buffer.iteration);
        }

        buffer_full = false; //buffer is safe for writing now
        #pragma omp flush(buffer_full)
      }// while true
    } /* End of disk write section. */
    /**************************************************************************/
  } // pragma parallel sections

  _mm_free(buffer.data);
  //--------------------------------------------------------------------------//
  //---------------------------- END OF YOUR CODE ----------------------------//
  //--------------------------------------------------------------------------// 
  
  double totalTime = omp_get_wtime() - elapsedTime;
  
  if (!parameters.batchMode)
    printf("\nExecution time of parallel (mode 2) version %.5f\n", totalTime);
  else
    printf("%s;%s;%f;%e;%e\n", outputFileName.c_str(), "par2", 
                               middleColAvgTemp, totalTime, 
                               totalTime / parameters.nIterations);   

  //-------------------- stop the stop watch  --------------------------------//

  // Close the output file
  if (file_id != H5I_INVALID_HID) H5Fclose(file_id);

  // return correct results in the correct array
  if (parameters.nIterations & 1)
  {
    memcpy(parResult, tempArray, materialProperties.nGridPoints * sizeof(float));
  }

  _mm_free(tempArray);
}// end of ParallelHeatDistribution
//------------------------------------------------------------------------------
  


/**
 * Store time step into output file (as a new dataset in Pixie format
 * @param [in] h5fileID  - handle to the output file
 * @param [in] Data      - data to write
 * @param [in] edgeSize  - size of the domain
 * @param [in] iteration - id of iteration);
 */
void StoreDataIntoFile(hid_t         h5fileId,
                       const float * data,
                       const size_t  edgeSize,
                       const size_t  snapshotId,
                       const size_t  iteration)
{
  hid_t    dataset_id, dataspace_id, group_id, attribute_id;
  hsize_t  dims[2] = {edgeSize, edgeSize};

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
                         H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
  H5Dwrite(dataset_id,
           H5T_NATIVE_FLOAT,H5S_ALL, H5S_ALL,H5P_DEFAULT,
           data);

  // close dataset
  H5Sclose(dataspace_id);


  // write attribute
  string atributeName="Time";
  dataspace_id = H5Screate(H5S_SCALAR);
  attribute_id = H5Acreate2 (group_id, atributeName.c_str(),
                             H5T_IEEE_F64LE, dataspace_id,
                             H5P_DEFAULT, H5P_DEFAULT);

  double snapshotTime = double(iteration);
  H5Awrite(attribute_id, H5T_IEEE_F64LE, &snapshotTime);
  H5Aclose(attribute_id);


  // Close the dataspace.
  H5Sclose(dataspace_id);

  // Close to the dataset.
  H5Dclose(dataset_id);
}// end of StoreDataIntoFile
//------------------------------------------------------------------------------





/**
 * Main function of the project
 * @param [in] argc
 * @param [in] argv
 * @return
 */
int main(int argc, char *argv[])
{

  ParseCommandline(argc, argv, parameters);

  // Create material properties and load from file
  TMaterialProperties materialProperties;
  materialProperties.LoadMaterialData(parameters.materialFileName);
  parameters.edgeSize = materialProperties.edgeSize;

  parameters.PrintParameters();

  // Memory allocation for output matrices.
  seqResult = (float*) _mm_malloc(materialProperties.nGridPoints * sizeof(float),
                                  DATA_ALIGNMENT);
  parResult = (float*) _mm_malloc(materialProperties.nGridPoints * sizeof(float),
                                  DATA_ALIGNMENT);

  // first touch for seq version
  for (size_t i = 0; i < materialProperties.nGridPoints; i++)
  {
    seqResult[i] = 0.0f;
  }

  // first touch policy
  #pragma omp parallel for
  for (size_t i = 0; i < materialProperties.nGridPoints; i++)
  {
    parResult[i] = 0.0f;
  }

  // run sequential version if needed
  if (parameters.IsRunSequntial())
  {
    SequentialHeatDistribution(seqResult,
                               materialProperties,
                               parameters,
                               parameters.outputFileName);
  }
  if (parameters.IsRunParallelNonOverlapped())
  {
    // run the parallel version with non-overlapped file output
    ParallelHeatDistributionNonOverlapped(parResult,
                                          materialProperties,
                                          parameters,
                                          parameters.outputFileName);
  }
  if (parameters.IsRunParallelOverlapped())
  {
    // run the parallel version with non-overlapped file output
    ParallelHeatDistributionOverlapped(parResult,
                                       materialProperties,
                                       parameters,
                                       parameters.outputFileName); 
  }
  
  // Validate the outputs
  if (parameters.IsValidation())
  {
    if (parameters.debugFlag)
    {
      printf("---------------- Sequential results ---------------\n");
      PrintArray(seqResult, materialProperties.edgeSize);

      printf("----------------- Parallel results ----------------\n");
      PrintArray(parResult, materialProperties.edgeSize);
    }

    if (VerifyResults(seqResult, parResult, parameters))
    {
      printf("Verification OK \n");
    }
    else
    {
      printf("Verification FAILED \n ");
    }
  }

  /* Memory deallocation*/
  _mm_free(seqResult);
  _mm_free(parResult);

  return EXIT_SUCCESS;
}// end of main
//------------------------------------------------------------------------------
