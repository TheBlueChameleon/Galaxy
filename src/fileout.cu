/* fileout.cu
 * 
 * this module provides methods to comfortably output the found configurations
 * to files.
 */

// ========================================================================= //
// dependencies

#include <stdio.h>
#include <stdlib.h>

#include "globals.h"
#include "fileout.h"

// ========================================================================= //
// global const vars

const char fn_galaxy[] = "./out/galaxy-coordinates.dat";

// ========================================================================= //
// prep and tidy up

bool fileout_initialized = false;

FILE * fh_galaxy;

// ========================================================================= //
// prep and tidy up

void quit_fileout();

void init_fileout() {
  fh_galaxy = fopen(fn_galaxy, "w");
  if (!fh_galaxy) {ABORT_WITH_MSG("could not open galaxy coordinates file.");}
  
  atexit(quit_fileout);
  fileout_initialized = true;
}

// ------------------------------------------------------------------------- //

void quit_fileout() {
  if (fh_galaxy) {fclose(fh_galaxy);}
  fileout_initialized = false;
}

// ========================================================================= //
// file writers

void fileout_galaxy() {
  fprintf(
    fh_galaxy,
    "# r.x\tr.y\tr.z\tv.x\tv.y\tv.z\tm\n"
  );
  
  for (unsigned int i=0; i<N_stars; i++) {
    fprintf(
      fh_galaxy,
      "%f\t%f\t%f\t%f\t%f\t%f\t%f\n",
      h_galaxy[i].position.x, h_galaxy[i].position.y, h_galaxy[i].position.z,
      h_galaxy[i].velocity.x, h_galaxy[i].velocity.y, h_galaxy[i].velocity.z,
      h_galaxy[i].mass
    );
  }
}
