/* fileout.h
 * 
 * this module provides methods to comfortably output the found configurations
 * to files.
 */

#ifndef LOGGER_H
#define LOGGER_H

// ========================================================================= //
// dependencies

#include <stdio.h>

#include "globals.h"

// ========================================================================= //
// global const vars

extern const char fn_galaxy[];

// ========================================================================= //
// global vars

extern bool fileout_initialized;

extern FILE * fh_galaxy;

// ========================================================================= //
// procs

void init_fileout();

void fileout_galaxy();

#endif//LOGGER_H
