/* TODO: Project definition
 */

#ifndef GLOBALS_H
#define GLOBALS_H

// ========================================================================= //
// dependencies

#include <cuda.h>
#include <curand.h>

// ========================================================================= //
// types 

typedef struct {
  float x;
  float y;
  float z;
} vector3D;

typedef struct {
  vector3D position;
  vector3D velocity;
  
  float mass;
} star;

// ========================================================================= //
// global const vars

extern const unsigned int N_stars     ;    // number of stars in the galaxy
extern const unsigned int N_datapoints;    // number of data points to report to host
extern const unsigned int N_steps     ;    // number of computations before copy-back is triggered
  // this means there are N_datapoints * N_steps computations before a reported quantity is copied back to host.

extern const unsigned int R_universe;      // initial coordinates will be within +/- R_universe for each coordinates component
extern const unsigned int M_star_max;      // maximum star mass, arbitrary units.


// ========================================================================= //
// device constants

__constant__ star *            GALAXY;
__constant__ curandGenerator_t RNG_MEM;

__constant__ unsigned int      N_STARS;
__constant__ unsigned int      R_UNIVERSE;
__constant__ unsigned int      M_STAR_MAX;

// ========================================================================= //
// global vars

extern bool flag_rand_initialized;

extern unsigned int blockSize;
extern unsigned int nBlocks;

extern star * d_galaxy;

extern curandGenerator_t d_RNG_mem;

// ========================================================================= //
// inline procs

// ----------------------------------------------------------------------- //
// CUDA error check interface

#define CudaCheckError()  cudaCheckErrorProc(__FILE__, __LINE__)
inline void cudaCheckErrorProc (const char * file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  
  if (err != cudaSuccess) {
    fprintf ( stderr, 
              "cudaCheckError() failed at %s :%i : %s\n",
              file, line, cudaGetErrorString(err) 
            );
            exit(-1);
  }

  err = cudaDeviceSynchronize ();
  if (err != cudaSuccess) {
    fprintf ( stderr, 
              "cudaCheckError() with sync failed at %s :%i : %s\n",
              file, line, cudaGetErrorString(err) 
            );
            exit(-1);
  }
#endif
  
  return;
}

// ----------------------------------------------------------------------- //
// host RNG interface -- if ever needed...

inline float randfloat  (                  ) {return (float) rand() / (float) RAND_MAX;}
inline float randbetween(float lo, float hi) {return (hi - lo) * randfloat() + lo;}

// ========================================================================= //
// procs

void init();

// ========================================================================= //
// macros


#endif//GLOBALS_H
