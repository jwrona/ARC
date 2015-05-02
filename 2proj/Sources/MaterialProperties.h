/**
 * @file        MaterialProperties.h
 * @author      Jiri Jaros and Vojtech Nikl\n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The header file with the medium properties
 *
 * @version     2015
 * @date        19 February 2015, 16:22 (created) \n
 *              26 February 2015, 16:22 (revised)
 *
 * @detail
 * This header file defines the class for material properties
 */

#ifndef MATERIAL_PROPERTIES_H
#define	MATERIAL_PROPERTIES_H

#include <string>
using namespace std;

/**
 * @class MaterialProperties
 * @brief This class maintains all medium parameters
 */
class TMaterialProperties
{
 public:
  /// Constructor
  TMaterialProperties() {};

  /// Destructor (free memory).
  ~TMaterialProperties();

  /// Load data from file
  void LoadMaterialData(const string fileName, bool loadData);

  /// Domain map - defines the type of the material at every gridpoint
  /// 0-air, 1-aluminum , 2-copper
  int   *domainMap;
  /// Thermal properties of the medium
  float *domainParams;
  /// Initial temperature distribution in the domain
  float *initTemp;
  /// Temperature of the air
  float  coolerTemp;
  /// Temperature of the heater
  float  heaterTemp;
  /// Size of the domain
  size_t edgeSize;
  /// Number of grid points
  size_t nGridPoints;

 private:

  /// Copy constructor is not allowed
  TMaterialProperties(const TMaterialProperties& orig);
  /// Operator = not allowed
  TMaterialProperties & operator = (const TMaterialProperties& orig);

};// end of MaterialProperties
//------------------------------------------------------------------------------

#endif	/* MATERIAL_PROPERTIES_H */

