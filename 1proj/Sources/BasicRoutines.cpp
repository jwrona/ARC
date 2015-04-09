/**
 * @file        BasicRoutines.cpp
 * @author      Jiri Jaros and Vojtech Nikl\n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file with basic routines
 *
 * @version     2015
 * @date        02 March 2015, 14:09 (created) \n
 *              02 March 2015, 16:22 (revised)
 *
 * @detail
 * This implementation file with basic routines and simulation parameters
 */

#include <getopt.h>
#include <string>
#include <omp.h>
#include <cmath>

#include "BasicRoutines.h"
#include "MaterialProperties.h"



/**
 * Print parameters
 */
void TParameters::PrintParameters() const
{
  if (batchMode)
  {
    printf("%ld;%ld;%d;%ld;%2.6f;%s;", edgeSize, nIterations, nThreads,
                                         diskWriteIntensity, airFlowRate, 
                                         materialFileName.c_str());
  }
  else
  {
    printf(".......... Parameters of the simulation ...........\n");
    printf("Domain size:          %ldx%ld\n", edgeSize,edgeSize);
    printf("Number of iterations: %ld\n",     nIterations);
    printf("Number of threads:    %d\n",      nThreads);
    printf("Disk write intensity: %ld\n",     diskWriteIntensity);
    printf("Air flow rate       : %2.6f\n",   airFlowRate);
    printf("Input file name     : %s \n",     materialFileName.c_str());
    printf("Output file name    : %s \n",     outputFileName.c_str());
    printf("Mode                : %d \n",     mode);
    printf("...................................................\n\n");
  }
}// end of PrintParameters
//------------------------------------------------------------------------------


/**
 * Parsing arguments from command line
 * @param [in] argc
 * @param [in] argv
 * @param [out] parameters - returns filled struct
 */
void ParseCommandline(int argc, char *argv[], TParameters & parameters)
{
  int c;

  bool n_flag = false;
  bool w_flag = false;
  bool t_flag = false;
  bool i_flag = false;
  bool m_flag = false;

  while ((c = getopt (argc, argv, "n:t:w:a:dvi:o:bm:")) != -1)
  {
    switch (c)
    {
      case 'n':
        parameters.nIterations = atol(optarg);
	      n_flag = true;
        break;

      case 'w':
        parameters.diskWriteIntensity = atol(optarg);
	      w_flag = true;
        break;

      case 'a':
        parameters.airFlowRate = atof(optarg);
        break;

      case 't':
        parameters.nThreads = atol(optarg);
        omp_set_num_threads(parameters.nThreads);
        t_flag = true;
        break;

      case 'd':
        parameters.debugFlag = true;
        break;

      case 'm':
	      parameters.mode = atol(optarg);
        m_flag = true;
        break;

      case 'v':
	      parameters.verificationFlag = true;
        break;

      case 'i':
        i_flag = true;
        parameters.materialFileName.assign(optarg);
        break;

      case 'o':
        parameters.outputFileName.assign(optarg);
        break;
      
      case 'b':
        parameters.batchMode = true;
        break;  

      default:
        fprintf(stderr,"Wrong parameter!\n");
        PrintUsageAndExit();
    }
  }// while

  if (!(n_flag && t_flag && i_flag && w_flag && m_flag) || 
      !(parameters.mode >= 0 && parameters.mode <= 2))
  {
    PrintUsageAndExit();
  }
}// end of ParseCommandline
//------------------------------------------------------------------------------


/**
 * Verify the difference between two corresponding gridpoints in the sequential
 * and parallel version. Print possible problems
 * @param [in] tempsSeq
 * @param [in] tempsPar
 * @param [in] parameters
 * @param [in] epsilon
 * @return true if no problem found
 */
bool VerifyResults(const float      * seqResult,
                   const float      * parResult,
                   const TParameters  parameters,
                   const float        epsilon)
{
  for (size_t i = 0; i < parameters.edgeSize * parameters.edgeSize; i++)
  {
    if (fabs(parResult[i] - seqResult[i]) > epsilon)
    {
      printf("Error found at position -> difference: ");
      printf("[%ld, %ld] -> %e \n", i / parameters.edgeSize, i % parameters.edgeSize,
                                    parResult[i] - seqResult[i]);
      return false;
    }
  }// for

  return true;
}//end of VerifyResults
//------------------------------------------------------------------------------

/**
 * Print usage and exit.
 */
void PrintUsageAndExit()
{
  fprintf(stderr,"Usage: \n");
  fprintf(stderr,"Mandatory arguments: \n");
  fprintf(stderr,"  -m [0-2]    mode 0 - run sequential version"); 
  fprintf(stderr,"              mode 1 - run parallel version (non-overlapped output)\n");
  fprintf(stderr,"              mode 2 - run parallel version (overlapped output)\n"); 
  fprintf(stderr,"  -n number of iterations \n");
  fprintf(stderr,"  -w disk write intensity (how often)\n");
  fprintf(stderr,"  -t number of threads \n");
  fprintf(stderr,"  -i material hdf5 file \n");
 
  fprintf(stderr,"Optional arguments: \n");
  fprintf(stderr,"  -o output hdf5 file \n");
  fprintf(stderr,"  -a air flow rate (values within <0.5, 0.0001> make sense \n");
  fprintf(stderr,"  -d set debug mode (compare results from seq and par version and write them to cout) \n");
  fprintf(stderr,"  -v verification mode (compare results of seq and par version) \n");

  fprintf(stderr,"  -b batch mode - output data in CSV format \n");
  
  exit(EXIT_FAILURE);
}// end of print usage
//------------------------------------------------------------------------------


/**
 *  Print array content
 * @param [in] data
 * @param [im] size
 */
void PrintArray(const float* data, const size_t edgeSize)
{
  for (size_t i = 0; i < edgeSize; i++)
  {
    printf("[Row %ld]: ", i);
    for (size_t j = 0; j < edgeSize; j++)
    {
      printf("%e, ", data[i * edgeSize + j]);
    }
    printf("\n");
  }
}// end of PrintArray
//------------------------------------------------------------------------------
