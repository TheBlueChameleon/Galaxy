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

__global__ void makeGalaxyOnDevice_component() {
  /* assumes that all components of GALAXY have been initialized in the range
   * 0.0f..1.0f and rescales them according to the global constants
   */
  
  
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
   * useful link:
   *   https://docs.nvidia.com/cuda/curand/host-api-overview.html
   */
  
  if (!flag_rand_initialized) {ABORT_WITH_MSG("RNG not initialized.");}
  
  // run the RNG
  curandGenerateUniform(
    d_RNG_mem, 
    (float *) d_galaxy,                             // galaxy is of type star *. star is a struct of only floats. This cast is justifiable.
    N_stars * (sizeof(*d_galaxy) / sizeof(float))
  );
  CudaCheckError();
  
  makeGalaxyOnDevice_component<<<nBlocks, blockSize>>>();
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
    N_stars * sizeof(*h_galaxy),
    cudaMemcpyDeviceToHost
  );
  CudaCheckError();
}

// ========================================================================= //
// get all distances from a given star at index k

// ------------------------------------------------------------------------- //
// proc device
__global__ void makeDistanceComponent(unsigned int k) {
  /* computes the vector distance and euclidean norm of this vector distance
   * to a given star k.
   */
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < N_STARS) {
    DISTANCES[i].x = GALAXY[i].position.x - GALAXY[k].position.x;
    DISTANCES[i].y = GALAXY[i].position.y - GALAXY[k].position.y;
    DISTANCES[i].z = GALAXY[i].position.z - GALAXY[k].position.z;
    
    DISTANCES[i].l = LENGTH3D(DISTANCES[i].x, DISTANCES[i].y, DISTANCES[i].z);
  }
}

// ------------------------------------------------------------------------- //
// proc host
void makeDistanceVector(unsigned int k) {
  if (k > N_stars) {
    fprintf(
      stderr,
      "%s: Invalid index %u\n",
      __func__, k
    );
    return;
  }
  
  makeDistanceComponent<<<nBlocks, blockSize>>>(k);
  cudaDeviceSynchronize();
}

// ========================================================================= //
// get all distances from origin for all stars

// ------------------------------------------------------------------------- //
// proc device
__global__ void makeModulusVectorComponent(
  action_t action,
  float *  dst
) {
  /* computes the modulus for each star's position or velocity
   */
  
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < N_STARS) {
    switch(action) {
      case position_action :
        dst[i] = LENGTH3D(GALAXY[i].position.x, GALAXY[i].position.y, GALAXY[i].position.z);
        
      case velocity_action :
        dst[i] = LENGTH3D(GALAXY[i].velocity.x, GALAXY[i].velocity.y, GALAXY[i].velocity.z);
    }
  }
}

// ------------------------------------------------------------------------- //
// proc host
void makeModulusVector (action_t action) {
  makeModulusVectorComponent<<<nBlocks, blockSize>>>(action, d_moduli);
  cudaDeviceSynchronize();
  
  cudaMemcpy(
    h_moduli,
    d_moduli, 
    N_stars * sizeof(*h_moduli),
    cudaMemcpyDeviceToHost
  );
  CudaCheckError();
}

// ========================================================================= //
// recenter galaxy in centre of mass and make average velocity = (0,0,0)

// ------------------------------------------------------------------------- //
// proc device

__global__ void copyWeightedComponent(
  action_t     action,
  vector3D_t * dst
) {
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < N_STARS) {
    switch(action) {
      case position_action :
        dst[i].x = GALAXY[i].position.x * GALAXY[i].mass / N_STARS;
        dst[i].y = GALAXY[i].position.y * GALAXY[i].mass / N_STARS;
        dst[i].z = GALAXY[i].position.z * GALAXY[i].mass / N_STARS;
        
      case velocity_action :
        dst[i].x = GALAXY[i].velocity.x * GALAXY[i].mass / N_STARS;
        dst[i].y = GALAXY[i].velocity.y * GALAXY[i].mass / N_STARS;
        dst[i].z = GALAXY[i].velocity.z * GALAXY[i].mass / N_STARS;
    }
  }
}
// ......................................................................... //
__global__ void reduction_galaxyCentre(
  action_t     action,
  vector3D_t * dst
  
) {
  
}
// ......................................................................... //
__global__ void translateComponent(
  action_t   action,
  vector3D_t offset
) {
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < N_STARS) {
    switch(action) {
      case position_action :
        GALAXY[i].position.x += offset.x;
        GALAXY[i].position.y += offset.y;
        GALAXY[i].position.z += offset.z;
        
      case velocity_action :
        GALAXY[i].velocity.x += offset.x;
        GALAXY[i].velocity.y += offset.y;
        GALAXY[i].velocity.z += offset.z;
    }
  }
}

// ------------------------------------------------------------------------- //
// proc host
void makeCentered() {
  /* This drives a reduction as for re-centering, the centre is needed, obviously.
   * Reduction is run on a copy of the galaxy's position and velocity coordinates.
   * 
   * In the following comments, COM stands for centre of mass, while COP 
   * represents centre of momentum.
   */
  
  vector3D_t  centreOfMass, 
              centreOfMomentum,
              * d_positions  = nullptr,     // temp device arrays 
              * d_velocities = nullptr;     // do reductions on these
  
  
  // get memory for reduction to COM & COP
  cudaMalloc(&d_positions , N_stars * sizeof(*d_positions));
  if (!d_positions ) {ABORT_WITH_MSG("position reduction device memory not initialized.");}
  CudaCheckError();
  
  cudaMalloc(&d_velocities, N_stars * sizeof(*d_velocities));
  if (!d_velocities) {ABORT_WITH_MSG("velocity reduction device memory not initialized.");}
  CudaCheckError();
  
  
  // copy to buffer COM & COP with weight mass[i]/N_stars
  copyWeightedComponent<<<nBlocks, blockSize>>>(position_action, d_positions );
  copyWeightedComponent<<<nBlocks, blockSize>>>(velocity_action, d_velocities);
  
  
  // get COM & COP
  // sync
  
  
  // free buffer COM & COP
  if (d_positions ) {cudaFree(d_positions ); CudaCheckError();}
  if (d_velocities) {cudaFree(d_velocities); CudaCheckError();}
  
  
  // translate by COM & COP
  translateComponent<<<nBlocks, blockSize>>>(position_action, centreOfMass    );
  translateComponent<<<nBlocks, blockSize>>>(velocity_action, centreOfMomentum);
}
