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
void makeModulusVector (action_t action);     // prepares global d_moduli and h_moduli

void makeCentered();                          // finds the centre of mass and mean velocity. Then translates the whole galaxy by these vectors.

#endif//STARDISTRIBUTION_H
