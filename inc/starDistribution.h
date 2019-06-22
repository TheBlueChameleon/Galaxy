/* starDistribution.h
 * 
 * this module deals with snapshots of the galaxy: generation, retrieval of 
 * data from device and reduction into radial histograms.
 */

#ifndef STARDISTRIBUTION_H
#define STARDISTRIBUTION_H

// ========================================================================= //
// dependencies

#include "globals.h"

// ========================================================================= //
// procs

void makeGalaxyOnDevice();
void fetchGalaxyFromDevice();

void makeDistanceVector(unsigned int i);

#endif//STARDISTRIBUTION_H
