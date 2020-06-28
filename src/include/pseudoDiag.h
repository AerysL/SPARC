/**
 * @file    pseudoDiag.h
 * @brief   This file contains the function declarations for Pulay sweep pseudo diagonalization. 
 *
 * @authors Shikhar Shah <sshikhar@gatech.edu>
 *          Hua Huang    <huangh223@gatech.edu>
 *          Edmond Chow  <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Edmond Group, Georgia Tech.
 * 
 * Reference: DOI 10.1002/jcc.540030214
 *            DOI 10.1137/0610025
 */

#ifndef __PSEUDO_DIAG_H__
#define __PSEUDO_DIAG_H__

// Pseudo-solve generalized eigenproblem H * x = lambda * M * x for subspace rotation
// Input parameters:
//   Ns   : Number of states, == the size of H and M matrices
//   H, M : Size Ns * Ns, matrix pencil of the eigenproblem (H and M are symmetric)
//   occ  : Size Ns, orbital occupation, range from 0 to 1
// Output parameters:
//   H, M   : These two matrices will be modified in pulay_dsygv
//   eigval : Size Ns, pseudo eigenvalues
//   eigvec : Size Ns * Ns, column-major, each column is a pseudo eigenvector
void pulay_dsygv(int Ns, double *H, double *M, double *eigval, double *eigvec, double *occ);

#endif

