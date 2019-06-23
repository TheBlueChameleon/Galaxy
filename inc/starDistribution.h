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

void makeGalaxyOnDevice();                    // prepares global d_galaxy
void fetchGalaxyFromDevice();                 // transfers to    h_galaxy

void makeDistanceVector(unsigned int i);      // prepares global d_distances
void makeRadiusVector  ();
void makeVelocityVector();

#endif//STARDISTRIBUTION_H
