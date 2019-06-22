/* TODO: Project definition
 */

// ========================================================================= //
// dependencies

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>   // srand
#include <time.h>

#include <cuda.h>
#include <curand.h>

#include "globals.h"

// ========================================================================= //
// global const vars

const unsigned int N_stars      = 1000000;    // number of stars in the galaxy
const unsigned int N_datapoints =     100;    // number of data points to report to host
const unsigned int N_steps      =     100;    // number of computations before copy-back is triggered
  // this means there are N_datapoints * N_steps computations before a reported quantity is copied back to host.

const unsigned int R_universe   = 10;         // initial coordinates will be within +/- R_universe for each coordinates component
const unsigned int M_star_max   = 10;         // maximum star mass, arbitrary units.

// ========================================================================= //
// global vars

bool flag_rand_initialized = false;

unsigned int blockSize = 256;
unsigned int nBlocks = (N_stars + blockSize - 1) / blockSize;

star * d_galaxy = nullptr;

curandGenerator_t d_RNG_mem;

// ========================================================================= //
// prep and tidy up

void quit();

void init() {
  /* This allocates memory for global objects such as the RNG or the galaxy.
   * Likewise it registers an exit handler that will free these ressources.
   */
  
  // RNG : host and device
  curandCreateGenerator(&d_RNG_mem, CURAND_RNG_PSEUDO_DEFAULT);
  CudaCheckError();
  
  auto seed = time(NULL);
  srand(seed);
  curandSetPseudoRandomGeneratorSeed(d_RNG_mem, seed);
  CudaCheckError();
  
  printf("Simulation will run on seed value %lu\n", seed);
  flag_rand_initialized = true;
  
  
  // get device memory for galaxy
  cudaMalloc(&d_galaxy, N_stars * sizeof(star));
  CudaCheckError();
  
  
  // set device constants
  cudaMemcpyToSymbol(GALAXY    , &d_galaxy  , sizeof(GALAXY    ));
  cudaMemcpyToSymbol(RNG_MEM   , &d_RNG_mem , sizeof(RNG_MEM   ));
  cudaMemcpyToSymbol(N_STARS   , &N_stars   , sizeof(N_STARS   ));
  cudaMemcpyToSymbol(R_UNIVERSE, &R_universe, sizeof(R_UNIVERSE));
  cudaMemcpyToSymbol(M_STAR_MAX, &M_star_max, sizeof(M_STAR_MAX));
  CudaCheckError();
  
  
  // make sure all of the above ressources will be free'd.
  atexit(quit);
}

// ----------------------------------------------------------------------- //

void quit() {
  curandDestroyGenerator(d_RNG_mem);
  CudaCheckError();
  
  cudaFree(d_galaxy);
  CudaCheckError();
}

// ========================================================================= //
// CUDA error


