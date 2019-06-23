/* globals.cu
 * 
 * this module defines the basic data types, global values and constants as
 * well as the procs generating and maintaining them (maintainance mostly means
 * automatically freeing ressources after use)
 */

// ========================================================================= //
// dependencies

//#include <assert.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "globals.h"

// ========================================================================= //
// global const vars

const unsigned int N_stars      = 256; //1000000;    // number of stars in the galaxy
const unsigned int N_datapoints =     100;    // number of data points to report to host
const unsigned int N_steps      =     100;    // number of computations before copy-back is triggered
  // this means there are N_datapoints * N_steps computations before a reported quantity is copied back to host.

const unsigned int D_universe   = 10;         // initial coordinates will be spread over +/- 1/2 D_universe for each coordinates component
const unsigned int V_init_max   =  1;         // as D_universe, but for the velocities.
const unsigned int M_star_max   = 10;         // maximum star mass, arbitrary units.

// ========================================================================= //
// device constants

__constant__ star_t     *      GALAXY;
__constant__ distance_t *      DISTANCES;

__constant__ unsigned int      N_STARS;
__constant__ unsigned int      D_UNIVERSE;
__constant__ unsigned int      M_STAR_MAX;
__constant__ unsigned int      V_INIT_MAX;

// ========================================================================= //
// global vars

bool flag_rand_initialized = false;

unsigned int blockSize = 256;
unsigned int nBlocks = (N_stars + blockSize - 1) / blockSize;

star_t     * d_galaxy    = nullptr;
star_t     * h_galaxy    = nullptr;

distance_t * d_distances = nullptr;
float      * d_moduli    = nullptr;

curandGenerator_t d_RNG_mem;

// ========================================================================= //
// prep and tidy up

void quit_globals();

void init_globals() {
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
  h_galaxy = (star_t *) malloc(N_stars * sizeof(star_t));
  if (!h_galaxy) {ABORT_WITH_MSG("galaxy host memory not initialized.");}
  
  cudaMalloc(&d_galaxy,      N_stars * sizeof(star_t));
  if (!d_galaxy) {ABORT_WITH_MSG("galaxy device memory not initialized.");}
  CudaCheckError();
  
  
  // get memory for distance and moduli vectors
  cudaMalloc(&d_distances, N_stars * sizeof(distance_t));
  if (!d_distances) {ABORT_WITH_MSG("distance device memory not initialized.");}
  CudaCheckError();
  
  cudaMalloc(&d_moduli, N_stars * sizeof(distance_t));
  if (!d_moduli) {ABORT_WITH_MSG("modulus device memory not initialized.");}
  CudaCheckError();
  
  
  // set device constants
  cudaMemcpyToSymbol(GALAXY    , &d_galaxy   , sizeof(GALAXY    ));
  cudaMemcpyToSymbol(DISTANCES , &d_distances, sizeof(DISTANCES ));
  
  cudaMemcpyToSymbol(N_STARS   , &N_stars    , sizeof(N_STARS   ));
  cudaMemcpyToSymbol(D_UNIVERSE, &D_universe , sizeof(D_UNIVERSE));
  cudaMemcpyToSymbol(M_STAR_MAX, &M_star_max , sizeof(M_STAR_MAX));
  cudaMemcpyToSymbol(V_INIT_MAX, &V_init_max , sizeof(V_INIT_MAX));
  CudaCheckError();
  
  
  // make sure all of the above ressources will be free'd.
  atexit(quit_globals);
}

// ------------------------------------------------------------------------- //

void quit_globals() {
  if (d_RNG_mem) {curandDestroyGenerator(d_RNG_mem); CudaCheckError();}
  flag_rand_initialized = false;
  
  if (h_galaxy) {free    (h_galaxy);                  }
  if (d_galaxy) {cudaFree(d_galaxy); CudaCheckError();}
}

// ========================================================================= //
// CUDA error


