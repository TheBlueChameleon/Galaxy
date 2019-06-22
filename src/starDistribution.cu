/* TODO: Project definition
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
// proc host

void d_makeStarsOnDevice() {
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
  
  if (!flag_rand_initialized) {
    fprintf ( stderr, 
              "d_makeStarsOnDevice: RNG not initialized. Aborting.\n"
            );
    exit(-1);
  }
  
  if (!d_galaxy) {
    fprintf ( stderr, 
              "d_makeStarsOnDevice: galaxy memory not initialized. Aborting.\n"
            );
    exit(-1);
  }
  
  
  // run the RNG
  curandGenerateUniform(
    d_RNG_mem, 
    (float *) d_galaxy,                             // galaxy is of type star *. star is a struct of only floats. This cast is justifiable.
    N_stars * (sizeof(star) / sizeof(float))
  );
  CudaCheckError();
  
  
}


// ========================================================================= //
// proc device

__global__ void galaxy_scale(
  star * galaxy, 
  const unsigned int N
) {
  int i = threadIdx.x;  //blockDim.x ∗ blockIdx.x + threadIdx.x;
  
  if (i < N) {
    galaxy[i].position.x = 0;//RANDBETWEEN(-R_universe, R_universe);
    galaxy[i].position.y = 0;//RANDBETWEEN(-R_universe, R_universe);
    galaxy[i].position.z = 0;//RANDBETWEEN(-R_universe, R_universe);
    
    galaxy[i].velocity.x = 0;
    galaxy[i].velocity.y = 0;
    galaxy[i].velocity.z = 0;
    
    galaxy[i].mass = 0;//M_star_max * RANDFLOAT;
  }
}
