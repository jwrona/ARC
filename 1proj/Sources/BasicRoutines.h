/**
 * @file        BasicRoutines.h
 * @author      Jiri Jaros and Vojtech Nikl\n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file with basic routines
 *
 * @version     2015
 * @date        02 March 2015, 14:09 (created) \n
 *              02 March 2015, 16:22 (revised)
 *
 * @detail
 * This header file with basic routines and simulation parameters
 */


#ifndef BASIC_ROUTINES_H
#define	BASIC_ROUTINES_H

#include <string>


#ifdef __AVX__
/**
 * @var DATA_ALIGNMENT
 * @brief memory alignment for AVX(32B)
 */
const int DATA_ALIGNMENT = 32;
#else

/**
 * @var DATA_ALIGNMENT
 * @brief memory alignment for SSE, SSE2, SSE3, SSE4 (16B)
 */const int DATA_ALIGNMENT = 16;
#endif



/**
 * @struct TParameters
 * Parameters of the algorithm
 */
struct TParameters
{
  /// Number of iterations of the main heat distr. algorithm
  size_t  nIterations;
  /// Number of measured points in one dimension
  size_t edgeSize;
  /// Number of OMP threads
  int    nThreads;
  /// Every Nth iteration goes into the HDF5 out file
  size_t diskWriteIntensity;
  /// air flow rate (how fast the air flows)
  float  airFlowRate;
  /// File name with the domain material properties
  std::string materialFileName;
  /// HDF5 file with the result to be visualized
  std::string outputFileName;

  /// Mode 0 - seq, 1 - par with no file output, 2 - par with file output
  int mode;
  ///  Compare results of sequential and parallel version
  bool debugFlag;
  /// Verify the result
  bool verificationFlag;
  /// If true, run both sequential and parallel version and compare times, otherwise run only parallel
  bool sequentialFlag;
  /// If true, output all data in CSV format
  bool batchMode;

  /// Default constructor
  TParameters() :
    nIterations(100000), edgeSize(0), nThreads(1),
    diskWriteIntensity(1000), airFlowRate(0.001f), 
    materialFileName(""), outputFileName(""), mode(0),
    debugFlag(false), verificationFlag(false), sequentialFlag(false), 
    batchMode(false) 
  {};

  /// run the sequential version?
  bool IsRunSequntial() const
  {
    return (verificationFlag ||debugFlag || mode == 0);
  }
  
  /// run parallel version with non-overlapped file output
  bool IsRunParallelNonOverlapped() const
  {
    return (mode == 1);
  }
  
  /// run parallel version with overlapped file output
  bool IsRunParallelOverlapped() const
  {
    return (mode == 2);
  }

  /// run data validation?
  bool IsValidation() const
  {
   return (debugFlag || verificationFlag);
  }

  void PrintParameters() const;

}; // end of TParameters
//------------------------------------------------------------------------------


/// Parse command line
void ParseCommandline(int argc, char *argv[], TParameters & parameters);

/// Print sage and exit
void PrintUsageAndExit();


/// Print array content
void PrintArray(const float * data,
                const size_t  edgeSize);

/// Verify the results between the sequential and parallel version
bool VerifyResults(const float * seqResult,
                   const float * parResult,
                   const TParameters parameters,
                   const float  epsilon = 0.001f);


#endif	/* BASICROUTINES_H */

