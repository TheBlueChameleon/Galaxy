/* TODO: Project definition
 */


// ========================================================================= //
// behaviour flags

#define CUDA_ERROR_CHECK

// ========================================================================= //
// dependencies


#include <stdio.h>
#include <assert.h>

#include "globals.h"
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
  
  init();
  
  
  printf("Attempting to run galaxy simulation. Runtime Parameters:\n");
  printf("   #stars            : %d\n", N_stars);
  printf("   #blocks           : %d\n", nBlocks);
  printf("   #threads per block: %d\n", blockSize);
  printf("\n");
  
  printf("Generating uniform star distribution on device...");
  d_makeStarsOnDevice();
  printf("done.\n");
  
  // ----------------------------------------------------------------------- //
  // tidy up
  
  return 0;
}
