/* starDistribution.cu
 * 
 * this module deals with snapshots of the galaxy: generation, retrieval of 
 * data from device and reduction into radial histograms.
 */


// ========================================================================= //
// dependencies

#include <ctime>
#include <stdio.h>

#include <cuda.h>
#include <curand.h>

#include "globals.h"
#include "starDistribution.h"


// ========================================================================= //
// make galaxy

// ------------------------------------------------------------------------- //
// proc device

__global__ void galaxy_scale() {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < N_STARS) {
    GALAXY[i].position.x -= 0.5;
    GALAXY[i].position.y -= 0.5;
    GALAXY[i].position.z -= 0.5;
    
    GALAXY[i].position.x *= D_UNIVERSE;
    GALAXY[i].position.y *= D_UNIVERSE;
    GALAXY[i].position.z *= D_UNIVERSE;
    
    GALAXY[i].velocity.x -= 0.5;
    GALAXY[i].velocity.y -= 0.5;
    GALAXY[i].velocity.z -= 0.5;
    
    GALAXY[i].velocity.x *= V_INIT_MAX;
    GALAXY[i].velocity.z *= V_INIT_MAX;
    GALAXY[i].velocity.z *= V_INIT_MAX;
    
    GALAXY[i].mass *= M_STAR_MAX;
  }
}

// ------------------------------------------------------------------------- //
// proc host

void makeGalaxyOnDevice() {
  /* This will initialize a galaxy with uniform distribution of
   * - position
   * - velocity
   * - mass
   * of the stars. This proc will place values 0.0f..1.0f into the given fields
   * and makes use of the below kernel to rescale/translate them into a 
   * reasonable range.
   * 
   * https://docs.nvidia.com/cuda/curand/host-api-overview.html
   */
  
  if (!flag_rand_initialized) {ABORT_WITH_MSG("RNG not initialized.");}
  
  // run the RNG
  curandGenerateUniform(
    d_RNG_mem, 
    (float *) d_galaxy,                             // galaxy is of type star *. star is a struct of only floats. This cast is justifiable.
    N_stars * (sizeof(star) / sizeof(float))
  );
  CudaCheckError();
  
  galaxy_scale<<<nBlocks, blockSize>>>();
  cudaDeviceSynchronize();
}

// ========================================================================= //
// get back galaxy from device

void fetchGalaxyFromDevice() {
  /* TODO: make this asynchronous
   * for this, you'll need to transform h_galaxy to a page locked array
   * use cudaMallocHost, cf. script p.33f.
   * 
   * This assumes that h_galaxy and d_galaxy have been properly initialized.
   * The "CTor" init() in globals.cu takes care of this.
   */
  
  cudaMemcpy(
    h_galaxy, 
    d_galaxy, 
    N_stars * sizeof(star),
    cudaMemcpyDeviceToHost
  );
  CudaCheckError();
}

// ------------------------------------------------------------------------- //
