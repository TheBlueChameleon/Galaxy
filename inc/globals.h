/* globals.h
 * 
 * this module defines the basic data types, global values and constants as
 * well as the procs generating and maintaining them (maintainance mostly means
 * automatically freeing ressources after use)
 * 
 * further, there are macros that are considered of global utility.
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
} vector3D_t;
// ......................................................................... //
typedef struct {
  vector3D_t position;
  vector3D_t velocity;
  float    mass;
} star_t;
// ------------------------------------------------------------------------- //
typedef struct {
  float x;
  float y;
  float z;
  float l;
} distance_t;
// ------------------------------------------------------------------------- //

// ========================================================================= //
// global const vars

extern const unsigned int N_stars     ;    // number of stars in the galaxy
extern const unsigned int N_datapoints;    // number of data points to report to host
extern const unsigned int N_steps     ;    // number of computations before copy-back is triggered
  // this means there are N_datapoints * N_steps computations before a reported quantity is copied back to host.

extern const unsigned int D_universe;      // initial coordinates will be within +/- D_universe for each coordinates component
extern const unsigned int M_star_max;      // maximum star mass, arbitrary units.
extern const unsigned int V_init_max;      // maximum initial star velocity in any canonical direction

// ========================================================================= //
// device constants

__constant__ extern star_t     *      GALAXY;
__constant__ extern distance_t *      DISTANCES;

__constant__ extern unsigned int      N_STARS;
__constant__ extern unsigned int      D_UNIVERSE;
__constant__ extern unsigned int      M_STAR_MAX;
__constant__ extern unsigned int      V_INIT_MAX;

// ========================================================================= //
// global vars

extern bool flag_rand_initialized;

extern unsigned int blockSize;
extern unsigned int nBlocks;

extern star_t     * d_galaxy;
extern star_t     * h_galaxy;

extern distance_t * d_distances;
extern float      * d_moduli;

extern curandGenerator_t d_RNG_mem;

// ========================================================================= //
// macros

#define CudaCheckError()  cudaCheckErrorProc(__FILE__, __LINE__)

#define ABORT_WITH_MSG(x) {                     \
    fprintf ( stderr,                           \
              "Abort in %s in %s, line %i\n",   \
              __func__, __FILE__, __LINE__      \
            );                                  \
    if (x) {                                    \
      fprintf ( stderr,                         \
                "\t%s\n",                       \
                x                               \
              );                                \
    }                                           \
    exit(-1);                                   \
  }

#define LENGTH3D(x, y, z) ((x)*(x) + (y)*(y) + (z)*(z))

#define DBG_VAL(fmt, x) {printf("%s = " fmt "\n", #x, x);}

// ========================================================================= //
// inline procs

// ------------------------------------------------------------------------- //
// CUDA error check interface

inline void cudaCheckErrorProc (const char * file, const int line) {
#ifdef CUDA_ERROR_CHECK
  cudaError err = cudaGetLastError();
  
  if (err != cudaSuccess) {ABORT_WITH_MSG(NULL);}

  err = cudaDeviceSynchronize ();
  if (err != cudaSuccess) {ABORT_WITH_MSG(NULL);}
#endif
}

// ------------------------------------------------------------------------- //
// host RNG interface -- if ever needed...

inline float randfloat  (                  ) {return (float) rand() / (float) RAND_MAX;}
inline float randbetween(float lo, float hi) {return (hi - lo) * randfloat() + lo;}

// ========================================================================= //
// procs

void init_globals();
#endif//GLOBALS_H
