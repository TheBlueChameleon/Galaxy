/* main.cu
 * 
 * this runs the main simulation
 */


// ========================================================================= //
// behaviour flags

#define CUDA_ERROR_CHECK

// ========================================================================= //
// dependencies


#include <stdio.h>
#include <assert.h>

#include "globals.h"
#include "fileout.h"
#include "starDistribution.h"

// ========================================================================= //
// kernels
 
// ========================================================================= //
// host procs

// ========================================================================= //
// main

int main () {
  // ----------------------------------------------------------------------- //
  // setup
  
  init_globals();
  init_fileout();
  
  
  printf("Attempting to run galaxy simulation. Runtime Parameters:\n");
  printf("   #stars            : %d\n", N_stars);
  printf("   #blocks           : %d\n", nBlocks);
  printf("   #threads per block: %d\n", blockSize);
  printf("\n");
  
  printf("Generating uniform star distribution on device...");
  makeGalaxyOnDevice();
  printf("done.\n");
  
  // ----------------------------------------------------------------------- //
  // make coordinates report
  
  printf("Fetching galaxy from device...");
  fetchGalaxyFromDevice();
  printf("done.\n");
  
  printf("writing galaxy coordinates file...");
  fileout_galaxy();
  printf("done.\n");
  
  // ----------------------------------------------------------------------- //
  // tidy up
  
  return 0;
}
