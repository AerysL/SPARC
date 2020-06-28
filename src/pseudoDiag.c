/**
 * @file    pseudoDiag.c
 * @brief   This file contains the function implementations for Pulay sweep pseudo diagonalization. 
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "pseudoDiag.h"

// BLAS and LAPACK routines
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

#define MAX(a, b) ((a)>(b))?(a):(b)
#define MIN(a, b) ((a)<(b))?(a):(b)

//#define OMP_PARALLEL
//#define FAST_SPARSE 

#ifdef OMP_PARALLEL
#include <omp.h>
#endif

// Perform a Jacobi rotation B = J^T * A * J where J = J(p, q, theta)
// Input parameters:
//   nrow : Number of rows and columns of A
//   p, q : Jacobi rotation index pair
//   A**  : Matrix A elements A(p, p), A(p, q), A(q, q)
//   G    : == V' * A, row-major, size >= ldA * nrow
//   ldG  : Leading dimension of G
//   V    : Eigenvectors, each row is a eigenvector
//   ldV  : Leading dimension of V
// Output parameters:
//   G, V : The p-th and q-th rows will be updated
static void jacobi_rotation_pair(
    const int nrow, const int p, const int q, 
    const double App, const double Apq, const double Aqq, 
    double *G, const int ldG, double *V, const int ldV
)
{
    double *Gp = G + p * ldG;
    double *Gq = G + q * ldG;
    double *Vp = V + p * ldV;
    double *Vq = V + q * ldV;
    
    // Calculate J = [c s;-s c] such that J' * Apq * J = diagonal
    double c, s, tau, t;
    if (fabs(Apq) < 1e-16)
    {
        c = 1.0; s = 0.0;
    } else {
        tau = (Aqq - App) / (2.0 * Apq);
        if (tau > 0) t =  1.0 / ( tau + sqrt(1.0 + tau * tau));
        else         t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
        c = 1.0 / sqrt(1.0 + t * t);
        s = t * c;
    }
    
    // Update G by applying J' on left
    // Update V by applying J  on right
    #pragma omp simd
    for (int l = 0; l < nrow; l++)
    {
        const double Gpl = Gp[l], Gql = Gq[l];
        const double Vpl = Vp[l], Vql = Vq[l];
        Gp[l] = c * Gpl - s * Gql;
        Gq[l] = s * Gpl + c * Gql;
        Vp[l] = c * Vpl - s * Vql;
        Vq[l] = s * Vpl + c * Vql;
    }
}

// Perform Jacobi rotations for pairs (p, q) in a block
// Input parameters:
//   nrow     : Number of rows and columns of A
//   thresh   : If Apq < thres, skip the rotation
//   blk_srow : Start row of the block
//   blk_nrow : Number of rows of the block
//   blk_scol : Start column of the block
//   blk_ncol : Number of columns of the block
//   G        : == V' * A, row-major, size >= ldA * nrow
//   ldG      : Leading dimension of G
//   V        : Cumulated Jacobi rotation matrix, each row is a eigenvector
//   ldV      : Leading dimension of V
// Output parameters:
//   G, V     : The p-th and q-th rows will be updated
//   <return> : If the rotation is performed
static int jacrot_blk_pair_thresh(
    const int nrow, const double thresh, const int blk_srow, const int blk_nrow, 
    const int blk_scol, const int blk_ncol, double *G, const int ldG, double *V, const int ldV
)
{
    int rot_cnt = 0;
    for (int p = blk_srow; p < blk_srow + blk_nrow; p++)
    {
        double *Gp = G + p * ldG;
        double *Vp = V + p * ldV;
        for (int q = blk_scol; q < blk_scol + blk_ncol; q++)
        {
            if (p >= q) continue;
            
            double *Gq = G + q * ldG;
            double *Vq = V + q * ldV;
            double App = 0.0, Aqq = 0.0, Apq = 0.0;
            #pragma omp simd
            for (int l = 0; l < nrow; l++)
                Apq += Gp[l] * Vq[l];
            if (fabs(Apq) < thresh) continue;

            #pragma omp simd
            for (int l = 0; l < nrow; l++)
            {
                App += Gp[l] * Vp[l];
                Aqq += Gq[l] * Vq[l];
            }
            rot_cnt++;
            jacobi_rotation_pair(nrow, p, q, App, Apq, Aqq, G, ldG, V, ldV);
        }
    }
    return rot_cnt;
}

static void copy_dbl_mat_blk(
    const double *src, const int lds, const int nrow, const int ncol,  
    double *dst, const int ldd
)
{
    for (int irow = 0; irow < nrow; irow++)
        memcpy(dst + irow * ldd, src + irow * lds, sizeof(double) * ncol);
}

static int jacrot_blk_thresh(
    const int nrow, const double thresh, const int blk_srow, const int blk_nrow, 
    const int blk_scol, const int blk_ncol, double *G, const int ldG, double *V, const int ldV,
    const int blk_size, double *workbuf
)
{
    int p_blk_spos = blk_srow;
    int q_blk_spos = blk_scol;
    int p_blk_size = blk_nrow;
    int q_blk_size = blk_ncol;
    int VAblk_size = p_blk_size + q_blk_size;

    double *G_p_blk = workbuf;
    double *G_q_blk = G_p_blk + blk_size * nrow;
    double *V_p_blk = G_q_blk + blk_size * nrow;
    double *V_q_blk = V_p_blk + blk_size * nrow;
    double *A_blk   = V_q_blk + blk_size * nrow;
    double *V_blk   = A_blk   + blk_size * blk_size * 4;

    double *G_p_blk_ = G + p_blk_spos * ldG;
    double *G_q_blk_ = G + q_blk_spos * ldG;
    double *V_p_blk_ = V + p_blk_spos * ldV;
    double *V_q_blk_ = V + q_blk_spos * ldV;
    
    // GT_blk_p = GT(:, blk_p_s:blk_p_e);
    copy_dbl_mat_blk(G_p_blk_, ldG, p_blk_size, nrow, G_p_blk, nrow);
    // GT_blk_q = GT(:, blk_q_s:blk_q_e);
    copy_dbl_mat_blk(G_q_blk_, ldG, q_blk_size, nrow, G_q_blk, nrow);
    // V_blk_p  =  V(:, blk_p_s:blk_p_e);
    copy_dbl_mat_blk(V_p_blk_, ldV, p_blk_size, nrow, V_p_blk, nrow);
    // V_blk_q  =  V(:, blk_q_s:blk_q_e);
    copy_dbl_mat_blk(V_q_blk_, ldV, q_blk_size, nrow, V_q_blk, nrow);

    int pp_offset = 0;
    int pq_offset = p_blk_size;
    int qp_offset = p_blk_size * VAblk_size;
    int qq_offset = p_blk_size * (VAblk_size + 1);
    
    // A_blk(s0:e0, s1:e1) = GT_blk_p' * V_blk_q;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, 
        p_blk_size, q_blk_size, nrow,
        1.0, G_p_blk, nrow, V_q_blk, nrow, 
        0.0, A_blk + pq_offset, VAblk_size
    );
    int rot_cnt = 0;
    for (int i = 0; i < p_blk_size; i++)
    {
        double *A_blk_i = A_blk + i * VAblk_size;
        for (int j = p_blk_size; j < VAblk_size; j++)
            if (fabs(A_blk_i[j]) >= thresh) rot_cnt++;
    }

    if (rot_cnt * 10 < p_blk_size * q_blk_size)
    {
        #ifdef FAST_SPARSE
        // Do it in the sparse way, not exactly the same as the old code
        for (int i = 0; i < p_blk_size; i++)
        {
            double *A_blk_i = A_blk + i * VAblk_size;
            for (int j = p_blk_size; j < VAblk_size; j++)
            {
                if (fabs(A_blk_i[j]) < thresh) continue;
                int p = blk_srow + i;
                int q = blk_scol + (j - p_blk_size);
                double *Gp = G + p * ldG;
                double *Vp = V + p * ldV;
                double *Gq = G + q * ldG;
                double *Vq = V + q * ldV;
                double App = 0.0, Aqq = 0.0, Apq = A_blk_i[j];
                #pragma omp simd
                for (int l = 0; l < nrow; l++)
                {
                    //Apq += Gp[l] * Vq[l];
                    App += Gp[l] * Vp[l];
                    Aqq += Gq[l] * Vq[l];
                }
                jacobi_rotation_pair(nrow, p, q, App, Apq, Aqq, G, ldG, V, ldV);
            }
        }
        #else
        // Do it in the sparse way, exactly the same as the old code
        rot_cnt = jacrot_blk_pair_thresh(
            nrow, thresh, blk_srow, blk_nrow, 
            blk_scol, blk_ncol, G, ldG, V, ldV
        );
        #endif
        return rot_cnt;
    }

    // A_blk(s1:e1, s0:e0) = GT_blk_q' * V_blk_p;
    // A_blk(s1:e1, s0:e0) = A_blk(s0:e0, s0:e0)';
    for (int i = p_blk_size; i < VAblk_size; i++)
    {
        double *A_blk_i0 = A_blk + i * VAblk_size;
        double *A_blk_0i = A_blk + i;
        for (int j = 0; j < p_blk_size; j++)
            A_blk_i0[j] = A_blk_0i[j * VAblk_size];
    }
    // A_blk(s0:e0, s0:e0) = GT_blk_p' * V_blk_p;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, 
        p_blk_size, p_blk_size, nrow,
        1.0, G_p_blk, nrow, V_p_blk, nrow, 
        0.0, A_blk + pp_offset, VAblk_size
    );
    // A_blk(s1:e1, s1:e1) = GT_blk_q' * V_blk_q;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, 
        q_blk_size, q_blk_size, nrow,
        1.0, G_q_blk, nrow, V_q_blk, nrow, 
        0.0, A_blk + qq_offset, VAblk_size
    );

    // V_blk = jacobi_block_sweep(A_blk, s0, e0, s1, e1);
    int srow1 = 0, nrow1 = p_blk_size, scol1 = p_blk_size, ncol1 = q_blk_size;
    memset(V_blk, 0, sizeof(double) * VAblk_size * VAblk_size);
    for (int i = 0; i < VAblk_size; i++) 
        V_blk[i * VAblk_size + i] = 1.0;
    rot_cnt = jacrot_blk_pair_thresh(
        VAblk_size, thresh, srow1, nrow1, scol1, ncol1,
        A_blk, VAblk_size, V_blk, VAblk_size
    );
    
    // Notice: V_blk in C is the transpose of V_blk in MATLAB
    // GT(:, blk_p_s:blk_p_e) = GT_blk_p * V_blk(s0:e0, s0:e0) + GT_blk_q * V_blk(s1:e1, s0:e0);
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        p_blk_size, nrow, p_blk_size, 
        1.0, V_blk + pp_offset, VAblk_size, G_p_blk, nrow, 
        0.0, G_p_blk_, ldG
    );
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        p_blk_size, nrow, q_blk_size, 
        1.0, V_blk + pq_offset, VAblk_size, G_q_blk, nrow, 
        1.0, G_p_blk_, ldG
    );
    // GT(:, blk_q_s:blk_q_e) = GT_blk_p * V_blk(s0:e0, s1:e1) + GT_blk_q * V_blk(s1:e1, s1:e1);
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        q_blk_size, nrow, p_blk_size, 
        1.0, V_blk + qp_offset, VAblk_size, G_p_blk, nrow, 
        0.0, G_q_blk_, ldG
    );
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        q_blk_size, nrow, q_blk_size, 
        1.0, V_blk + qq_offset, VAblk_size, G_q_blk, nrow, 
        1.0, G_q_blk_, ldG
    );
    // V(:, blk_p_s:blk_p_e)  =  V_blk_p * V_blk(s0:e0, s0:e0) +  V_blk_q * V_blk(s1:e1, s0:e0);
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        p_blk_size, nrow, p_blk_size, 
        1.0, V_blk + pp_offset, VAblk_size, V_p_blk, nrow, 
        0.0, V_p_blk_, ldV
    );
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        p_blk_size, nrow, q_blk_size, 
        1.0, V_blk + pq_offset, VAblk_size, V_q_blk, nrow, 
        1.0, V_p_blk_, ldV
    );
    // V(:, blk_q_s:blk_q_e)  =  V_blk_p * V_blk(s0:e0, s1:e1) +  V_blk_q * V_blk(s1:e1, s1:e1);
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        q_blk_size, nrow, p_blk_size, 
        1.0, V_blk + qp_offset, VAblk_size, V_p_blk, nrow, 
        0.0, V_q_blk_, ldV
    );
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        q_blk_size, nrow, q_blk_size, 
        1.0, V_blk + qq_offset, VAblk_size, V_q_blk, nrow, 
        1.0, V_q_blk_, ldV
    );

    return rot_cnt;
}

static void qsort_dbl_int_keyval(double *key, int *val, int l, int r)
{
    int i = l, j = r, tmp_val;
    double mid_key = key[(l + r) / 2], tmp_key;
    while (i <= j)
    {
        while (key[i] < mid_key) i++;
        while (key[j] > mid_key) j--;
        if (i <= j)
        {
            tmp_key = key[i]; key[i] = key[j]; key[j] = tmp_key;
            tmp_val = val[i]; val[i] = val[j]; val[j] = tmp_val;
            i++;  j--;
        }
    }
    if (i < r) qsort_dbl_int_keyval(key, val, i, r);
    if (j > l) qsort_dbl_int_keyval(key, val, l, j);
}

// Generate next set of pairs for elimination
// Ref: Matrix Computation 4th edition, page 482
// Input parameters:
//   top, bot : (top[k], bot[k]) is a pair
//   npair    : Total number of pairs to be eliminated
// Output parameters:
//   top, bot : New sets of pairs
static void next_elimination_pairs(int *top, int *bot, const int npair)
{
    int top_tail = top[npair - 1];
    int bot_head = bot[0];
    for (int l = npair - 1; l >= 2; l--) top[l] = top[l - 1];
    for (int l = 0; l < npair; l++) bot[l] = bot[l + 1];
    top[1] = bot_head;
    bot[npair - 1] = top_tail;
}

static int blk_spos(const int blk_size, const int blk_rem, const int iblock)
{
    int res;
    if (iblock < blk_rem) res = (blk_size + 1) * iblock;
    else res = blk_size * iblock + blk_rem;
    return res;
}

void pulay_dsygv(int Ns, double *H, double *M, double *eigval, double *eigvec, double *occ)
{
    double pul_s = MPI_Wtime();

    // 0. Reduce the generalized eigenproblem to standard eigenproblem
    double chol_s = MPI_Wtime();
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', Ns, M, Ns);
    double chol_f = MPI_Wtime();
    double trnH_s = MPI_Wtime();
    cblas_dtrsm(CblasColMajor, CblasLeft,  CblasUpper, CblasTrans,   CblasNonUnit, Ns, Ns, 1.0, M, Ns, H, Ns);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, Ns, Ns, 1.0, M, Ns, H, Ns);
    double trnH_f = MPI_Wtime();

    // 1. Find out occupied, partial occupied, and virtual orbitals
    int ostart = 0, pstart = 0, vstart = 0, p_buffer = 2;
    double p_thresh = 1e-6;
    double o_thresh = 1.0 - p_thresh;
    for (int ix = 1; ix < Ns; ix++)
    {
        if ((occ[ix - 1] >= o_thresh) && (occ[ix] < o_thresh)) pstart = ix;
        if ((occ[ix - 1] >= p_thresh) && (occ[ix] < p_thresh)) vstart = ix;
    }
    pstart = pstart - p_buffer;
    vstart = vstart + p_buffer;
    if (pstart < 0)        pstart = 0;
    if (vstart > (Ns - 1)) vstart = Ns - 1;
    int num_o   = pstart - ostart;
    int num_p   = vstart - pstart;
    int num_v   = Ns     - vstart;
    int max_rot = num_o * num_v + num_o * num_p + num_p * num_v + num_p * (num_p-1) / 2;

    // 2. Setup block size and multi-threading buffers
    int n_thread, blk_size, blk_size1, n_block, semi_n_block, blk_rem;
    #ifdef OMP_PARALLEL
    n_thread  = omp_get_max_threads();
    #else
    n_thread  = 1;
    #endif
    blk_size  = 48;
    n_block   = Ns / blk_size;
    n_block   = MAX(n_block, 1);
    n_block   = (n_block + 1) / 2 * 2;  // n_block need to be even
    blk_size  = Ns / n_block;
    blk_rem   = Ns % n_block;
    blk_size1 = (blk_rem > 0) ? (blk_size + 1) : blk_size;
    semi_n_block = n_block / 2;
    n_thread  = MIN(n_thread, semi_n_block);

    
    int *topbot = (int*) malloc(sizeof(int) * n_block);
    int *top = topbot;
    int *bot = topbot + semi_n_block;
    for (int i = 0; i < semi_n_block; i++)
    {
        top[i] = 2 * i;
        bot[i] = 2 * i + 1;
    }

    int workbuf_size = 4 * Ns * blk_size1 + 8 * blk_size1 * blk_size1;
    double *workbuf = (double*) malloc(sizeof(double) * workbuf_size * n_thread);

    // 3. Use Jacobi rotation to "eliminate" 4 blocks and construct eigenvectors.
    //    The order of blocks to be eliminated cannot be changed.

    int rot_cnt  = 0;
    double jacrot_thresh = 1e-6;
    double jac_s = MPI_Wtime();

    // V is the eigenvectors, each row of V is one eigenvector
    double *V = (double*) malloc(sizeof(double) * Ns * Ns);
    memset(V, 0, sizeof(double) * Ns * Ns);
    for (int i = 0; i < Ns; i++) V[i * Ns + i] = 1.0;

    // 3.1 Eliminate occupied-virtual block, OpenMP parallelized
    #ifdef OMP_PARALLEL
    #pragma omp parallel num_threads(n_thread) reduction(+:rot_cnt)
    {
        int tid = omp_get_thread_num();
        double *thread_workbuf = workbuf + tid * workbuf_size;

        // Off-diagonal blocks
        for (int subsweep = 0; subsweep < n_block - 1; subsweep++)
        {
            #pragma omp barrier
            #pragma omp for schedule(dynamic)
            for (int k = 0; k < semi_n_block; k++)
            {
                int blk_p = MIN(top[k], bot[k]);
                int blk_q = MAX(top[k], bot[k]);

                int blk_p_spos = blk_spos(blk_size, blk_rem, blk_p);
                int blk_q_spos = blk_spos(blk_size, blk_rem, blk_q);
                int blk_p_epos = blk_spos(blk_size, blk_rem, blk_p + 1);
                int blk_q_epos = blk_spos(blk_size, blk_rem, blk_q + 1);

                // Check if this block overlaps with the occupied-virtual block
                if (blk_p_spos >= pstart) continue;
                if (blk_q_epos <  vstart) continue;
                blk_p_epos = MIN(blk_p_epos, pstart);
                blk_q_spos = MAX(blk_q_spos, vstart);
                int blk_p_size = blk_p_epos - blk_p_spos;
                int blk_q_size = blk_q_epos - blk_q_spos;

                rot_cnt += jacrot_blk_thresh(
                    Ns, jacrot_thresh, blk_p_spos, blk_p_size,
                    blk_q_spos, blk_q_size, H, Ns, V, Ns, 
                    blk_size1, thread_workbuf
                );
            }

            #pragma omp master
            next_elimination_pairs(top, bot, semi_n_block);
        }

        // Diagonal blocks
        #pragma omp for schedule(dynamic)
        for (int k = 0; k < n_block; k++)
        {
            int blk_p_spos = blk_spos(blk_size, blk_rem, k);
            int blk_p_epos = blk_spos(blk_size, blk_rem, k + 1);
            int blk_q_spos = blk_p_spos;
            int blk_q_epos = blk_p_epos;

            // Check if this block overlaps with the occupied-virtual block
            if (blk_p_spos >= pstart) continue;
            if (blk_q_epos <  vstart) continue;
            blk_p_epos = MIN(blk_p_epos, pstart);
            blk_q_spos = MAX(blk_q_spos, vstart);
            int blk_p_size = blk_p_epos - blk_p_spos;
            int blk_q_size = blk_q_epos - blk_q_spos;

            rot_cnt += jacrot_blk_thresh(
                Ns, jacrot_thresh, blk_p_spos, blk_p_size,
                blk_q_spos, blk_q_size, H, Ns, V, Ns, 
                blk_size1, thread_workbuf
            );
        }
    }
    #else
    for (int i = ostart; i < pstart; i += blk_size)
    {
        int blk_size_i = (i + blk_size > pstart) ? (pstart - i) : blk_size;
        for (int j = vstart; j < Ns; j += blk_size)
        {
            int blk_size_j = (j + blk_size > Ns) ? (Ns - j) : blk_size;
            rot_cnt += jacrot_blk_thresh(
                Ns, jacrot_thresh, i, blk_size_i,
                j, blk_size_j, H, Ns, V, Ns, 
                blk_size, workbuf
            );
        }
    }
    #endif

    // 3.2 Eliminate occupied-partial block
    for (int i = ostart; i < pstart; i += blk_size)
    {
        int blk_size_i = (i + blk_size > pstart) ? (pstart - i) : blk_size;
        for (int j = pstart; j < vstart; j += blk_size)
        {
            int blk_size_j = (j + blk_size > vstart) ? (vstart - j) : blk_size;
            rot_cnt += jacrot_blk_thresh(
                Ns, jacrot_thresh, i, blk_size_i,
                j, blk_size_j, H, Ns, V, Ns, 
                blk_size, workbuf
            );
        }
    }

    // 3.3 Eliminate partial-partial  block 
    rot_cnt += jacrot_blk_pair_thresh(
        Ns, jacrot_thresh, pstart, num_p,  
        pstart, num_p, H, Ns, V, Ns
    );

    // 3.4 Eliminate partial-virtual  block
    for (int j = vstart; j < Ns; j += blk_size)
    {
        int blk_size_j = (j + blk_size > Ns) ? (Ns - j) : blk_size;
        for (int i = pstart; i < vstart; i += blk_size)
        {
            int blk_size_i = (i + blk_size > vstart) ? (vstart - i) : blk_size;
            rot_cnt += jacrot_blk_thresh(
                Ns, jacrot_thresh, i, blk_size_i,
                j, blk_size_j, H, Ns, V, Ns,
                blk_size, workbuf
            );
        }
    }
    
    double jac_f = MPI_Wtime();

    // 4. Sort eigenvalues in ascending order and rearrange eigenvectors
    //    eigvec also stores each eigenvector in a row, we don't need to transpose V
    double sor_s = MPI_Wtime();
    int *eigpos = (int*) malloc(sizeof(int) * Ns);
    for (int i = 0; i < Ns; i++)
    {
        double *H_i = H + i * Ns;
        double *V_i = V + i * Ns;
        eigpos[i] = i;
        double res = 0.0;
        #pragma omp simd
        for (int j = 0; j < Ns; j++)
            res += H_i[j] * V_i[j];
        eigval[i] = res;
    }
    qsort_dbl_int_keyval(eigval, eigpos, 0, Ns-1);
    for (int i = 0; i < Ns; i++)
        memcpy(eigvec + i * Ns, V + eigpos[i] * Ns, sizeof(double) * Ns);
    double sor_f = MPI_Wtime();

    // 5. Transform the eigenvectors for generalized eigenproblem
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, Ns, Ns, 1.0, M, Ns, eigvec, Ns);

    double pul_f = MPI_Wtime();

    free(eigpos);
    free(V);
    free(workbuf);
    free(topbot);

    #ifdef DEBUG
    printf("Pulay dsygv sub-timings:\n");
    printf("Total time : %.3lf ms\n", 1000.0*(pul_f - pul_s));
    printf("Cholesky   : %.3lf ms\n", 1000.0*(chol_f - chol_s));
    printf("Transform H: %.3lf ms\n", 1000.0*(trnH_f - trnH_s));
    printf("Jacobi rot : %.3lf ms\n", 1000.0*(jac_f - jac_s));
    printf("Sorting    : %.3lf ms\n", 1000.0*(sor_f - sor_s));
    printf("Jacobi rot %% : %.2lf %%\n", 100.0 * (double)rot_cnt / (double)max_rot);
    printf("\n");
    #endif
}

