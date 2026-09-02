#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  O
1  N
*/

// Species
#define O2_ID 0
#define N2_ID 1
#define NO_ID 2
#define N_ID 3
#define O_ID 4

#define NUM_GAS_ELEMENTS 2  // Elements in the homogeneous phase
#define NUM_GAS_SPECIES 5   // Species in the homogeneous phase
#define NUM_GAS_REACTIONS 5 // Reactions in the homogeneous phase

#define SITE_DENSITY 0.000000E+00 // mol/cm^2

#define NUM_SURFACE_ELEMENTS 0  // Additional elements in heterogeneous phase
#define NUM_SURFACE_SPECIES 0   // Species in the heterogeneous phase
#define NUM_SURFACE_REACTIONS 0 // Reactions in the heterogeneous phase

#define NUM_ELEMENTS (NUM_GAS_ELEMENTS + NUM_SURFACE_ELEMENTS)
#define NUM_SPECIES (NUM_GAS_SPECIES + NUM_SURFACE_SPECIES)
#define NUM_REACTIONS (NUM_GAS_REACTIONS + NUM_SURFACE_REACTIONS)

#define NUM_IONS 0

#define NUM_FIT 4

//  ALWAYS on CPU stuff -- can have different def depending on if we are CPU or
//  GPU based. Defined in mechanism.cpp MISC
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYMS_STR(amrex::Vector<std::string>& kname);
void GET_RMAP(int* _rmap);
void CKINU(const int i, int& nspec, int* ki, int* nu);
void CKKFKR(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real* x,
  amrex::Real* q_f,
  amrex::Real* q_r);
void progressRateFR(
  amrex::Real* q_f, amrex::Real* q_r, amrex::Real* sc, amrex::Real T);
//  SPARSE INFORMATION
void SPARSITY_INFO(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST_SIMPLIFIED(int* nJdata, const int* consP);
void
SPARSITY_PREPROC_CSC(int* rowVals, int* colPtrs, const int* consP, int NCELLS);
void SPARSITY_PREPROC_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(
  int* rowVals, int* colPtrs, int* indx, const int* consP);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(
  int* colVals, int* rowPtr, const int* consP, int base);

// A few mechanism parameters
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKINDX(int& mm, int& kk, int& ii, int& nfit)
{
  mm = 2;
  kk = 5;
  ii = 5;
  nfit = -1; // Why do you need this anyway ?
}

// save atomic weights into array
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
atomicWeight(amrex::Real* awt)
{
  awt[0] = 15.999400; // O
  awt[1] = 14.006700; // N
}

// get atomic weight for all elements
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKAWT(amrex::Real* awt)
{
  atomicWeight(awt);
}

// Returns the elemental composition
// of the speciesi (mdim is num of elements)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKNCF(int* ncf)
{
  int kd = 2;
  // Zero ncf
  for (int id = 0; id < kd * 5; ++id) {
    ncf[id] = 0;
  }

  // O2
  ncf[0 * kd + 0] = 2; // O

  // N2
  ncf[1 * kd + 1] = 2; // N

  // NO
  ncf[2 * kd + 1] = 1; // N
  ncf[2 * kd + 0] = 1; // O

  // N
  ncf[3 * kd + 1] = 1; // N

  // O
  ncf[4 * kd + 0] = 1; // O
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[5] = {
  3.1251171918946961e-02, // O2
  3.5697202053303061e-02, // N2
  3.3326556933423540e-02, // NO
  7.1394404106606121e-02, // N
  6.2502343837893923e-02, // O
};
#endif
const amrex::Real h_global_imw[5] = {
  3.1251171918946961e-02, // O2
  3.5697202053303061e-02, // N2
  3.3326556933423540e-02, // NO
  7.1394404106606121e-02, // N
  6.2502343837893923e-02, // O
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[5] = {
  3.1998799999999999e+01, // O2
  2.8013400000000001e+01, // N2
  3.0006100000000000e+01, // NO
  1.4006700000000000e+01, // N
  1.5999400000000000e+01, // O
};
#endif
const amrex::Real h_global_mw[5] = {
  3.1998799999999999e+01, // O2
  2.8013400000000001e+01, // N2
  3.0006100000000000e+01, // NO
  1.4006700000000000e+01, // N
  1.5999400000000000e+01, // O
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 3.1251171918946961e-02; // O2
  imw_new[1] = 3.5697202053303061e-02; // N2
  imw_new[2] = 3.3326556933423540e-02; // NO
  imw_new[3] = 7.1394404106606121e-02; // N
  imw_new[4] = 6.2502343837893923e-02; // O
}

//  inverse molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
imw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_imw[n];
#else
  return h_global_imw[n];
#endif
}
//  molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_mw(amrex::Real* mw_new)
{
  mw_new[0] = 3.1998799999999999e+01; // O2
  mw_new[1] = 2.8013400000000001e+01; // N2
  mw_new[2] = 3.0006100000000000e+01; // NO
  mw_new[3] = 1.4006700000000000e+01; // N
  mw_new[4] = 1.5999400000000000e+01; // O
}

//  molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
mw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_mw[n];
#else
  return h_global_mw[n];
#endif
}

//  Returns R, Rc, Patm
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRP(amrex::Real& ru, amrex::Real& ruc, amrex::Real& pa)
{
  ru = 8.31446261815324e+07;
  ruc = 1.98721558317399615845;
  pa = 1.01325e+06;
}

// compute Cv/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cv_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = +2.614600000000000e+00 - 1.859800000000000e-03 * T +
                 7.081400000000000e-06 * T2 - 6.807000000000000e-09 * T3 +
                 2.162800000000000e-12 * T4;
    // species 1: N2
    species[1] = +2.674800000000000e+00 - 1.208100000000000e-03 * T +
                 2.324000000000000e-06 * T2 - 6.321800000000000e-10 * T3 -
                 2.257700000000000e-13 * T4;
    // species 2: NO
    species[2] = +2.588700000000000e+00 - 1.247900000000000e-03 * T +
                 3.978600000000000e-06 * T2 - 2.865100000000000e-09 * T3 +
                 6.301500000000000e-13 * T4;
    // species 3: N
    species[3] = +1.503100000000000e+00 - 2.180000000000000e-05 * T +
                 5.420500000000000e-08 * T2 - 5.647600000000000e-11 * T3 +
                 2.099900000000000e-14 * T4;
    // species 4: O
    species[4] = +1.823600000000000e+00 - 8.947800000000000e-04 * T +
                 8.306000000000000e-07 * T2 - 1.683700000000000e-10 * T3 -
                 7.320500000000000e-14 * T4;
  } else {
    // species 0: O2
    species[0] = +2.594900000000000e+00 + 7.521300000000001e-04 * T -
                 1.873200000000000e-07 * T2 + 2.791300000000000e-11 * T3 -
                 1.577400000000000e-15 * T4;
    // species 1: N2
    species[1] = +2.212500000000000e+00 + 1.013700000000000e-03 * T -
                 3.046700000000000e-07 * T2 + 4.109100000000000e-11 * T3 -
                 2.017000000000000e-15 * T4;
    // species 2: NO
    species[2] = +2.204700000000000e+00 + 1.270500000000000e-03 * T -
                 4.660300000000000e-07 * T2 + 7.500699999999999e-11 * T3 -
                 4.231400000000000e-15 * T4;
    // species 3: N
    species[3] = +1.482000000000000e+00 + 6.925800000000001e-05 * T -
                 6.306500000000000e-08 * T2 + 1.838700000000000e-11 * T3 -
                 1.174700000000000e-15 * T4;
    // species 4: O
    species[4] = +1.542100000000000e+00 - 2.755100000000000e-05 * T -
                 3.102800000000000e-09 * T2 + 4.551100000000000e-12 * T3 -
                 4.368100000000000e-16 * T4;
  }
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cp_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = +3.614600000000000e+00 - 1.859800000000000e-03 * T +
                 7.081400000000000e-06 * T2 - 6.807000000000000e-09 * T3 +
                 2.162800000000000e-12 * T4;
    // species 1: N2
    species[1] = +3.674800000000000e+00 - 1.208100000000000e-03 * T +
                 2.324000000000000e-06 * T2 - 6.321800000000000e-10 * T3 -
                 2.257700000000000e-13 * T4;
    // species 2: NO
    species[2] = +3.588700000000000e+00 - 1.247900000000000e-03 * T +
                 3.978600000000000e-06 * T2 - 2.865100000000000e-09 * T3 +
                 6.301500000000000e-13 * T4;
    // species 3: N
    species[3] = +2.503100000000000e+00 - 2.180000000000000e-05 * T +
                 5.420500000000000e-08 * T2 - 5.647600000000000e-11 * T3 +
                 2.099900000000000e-14 * T4;
    // species 4: O
    species[4] = +2.823600000000000e+00 - 8.947800000000000e-04 * T +
                 8.306000000000000e-07 * T2 - 1.683700000000000e-10 * T3 -
                 7.320500000000000e-14 * T4;
  } else {
    // species 0: O2
    species[0] = +3.594900000000000e+00 + 7.521300000000001e-04 * T -
                 1.873200000000000e-07 * T2 + 2.791300000000000e-11 * T3 -
                 1.577400000000000e-15 * T4;
    // species 1: N2
    species[1] = +3.212500000000000e+00 + 1.013700000000000e-03 * T -
                 3.046700000000000e-07 * T2 + 4.109100000000000e-11 * T3 -
                 2.017000000000000e-15 * T4;
    // species 2: NO
    species[2] = +3.204700000000000e+00 + 1.270500000000000e-03 * T -
                 4.660300000000000e-07 * T2 + 7.500699999999999e-11 * T3 -
                 4.231400000000000e-15 * T4;
    // species 3: N
    species[3] = +2.482000000000000e+00 + 6.925800000000001e-05 * T -
                 6.306500000000000e-08 * T2 + 1.838700000000000e-11 * T3 -
                 1.174700000000000e-15 * T4;
    // species 4: O
    species[4] = +2.542100000000000e+00 - 2.755100000000000e-05 * T -
                 3.102800000000000e-09 * T2 + 4.551100000000000e-12 * T3 -
                 4.368100000000000e-16 * T4;
  }
}

// compute the g/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
gibbs(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = -1.044000000000000e+03 * invT - 7.482000000000002e-01 -
                 3.614600000000000e+00 * logT + 9.299000000000000e-04 * T -
                 1.180233333333333e-06 * T2 + 5.672500000000000e-10 * T3 -
                 1.081400000000000e-13 * T4;
    // species 1: N2
    species[1] = -1.043000000000000e+03 * invT + 1.316800000000000e+00 -
                 3.674800000000000e+00 * logT + 6.040500000000000e-04 * T -
                 3.873333333333333e-07 * T2 + 5.268166666666667e-11 * T3 +
                 1.128850000000000e-14 * T4;
    // species 2: NO
    species[2] = +9.764000000000000e+03 * invT - 1.561000000000000e+00 -
                 3.588700000000000e+00 * logT + 6.239500000000000e-04 * T -
                 6.631000000000001e-07 * T2 + 2.387583333333333e-10 * T3 -
                 3.150750000000000e-14 * T4;
    // species 3: N
    species[3] = +5.613000000000000e+04 * invT - 1.664500000000000e+00 -
                 2.503100000000000e+00 * logT + 1.090000000000000e-05 * T -
                 9.034166666666666e-09 * T2 + 4.706333333333333e-12 * T3 -
                 1.049950000000000e-15 * T4;
    // species 4: O
    species[4] = +2.915000000000000e+04 * invT - 6.791000000000000e-01 -
                 2.823600000000000e+00 * logT + 4.473900000000000e-04 * T -
                 1.384333333333333e-07 * T2 + 1.403083333333333e-11 * T3 +
                 3.660250000000000e-15 * T4;
  } else {
    // species 0: O2
    species[0] = -1.044000000000000e+03 * invT - 2.404000000000002e-01 -
                 3.594900000000000e+00 * logT - 3.760650000000000e-04 * T +
                 3.122000000000000e-08 * T2 - 2.326083333333333e-12 * T3 +
                 7.887000000000000e-17 * T4;
    // species 1: N2
    species[1] = -1.043000000000000e+03 * invT - 1.153600000000000e+00 -
                 3.212500000000000e+00 * logT - 5.068500000000000e-04 * T +
                 5.077833333333333e-08 * T2 - 3.424250000000000e-12 * T3 +
                 1.008500000000000e-16 * T4;
    // species 2: NO
    species[2] = +9.764000000000000e+03 * invT - 3.482000000000000e+00 -
                 3.204700000000000e+00 * logT - 6.352500000000000e-04 * T +
                 7.767166666666667e-08 * T2 - 6.250583333333333e-12 * T3 +
                 2.115700000000000e-16 * T4;
    // species 3: N
    species[3] = +5.613000000000000e+04 * invT - 1.779800000000000e+00 -
                 2.482000000000000e+00 * logT - 3.462900000000000e-05 * T +
                 1.051083333333333e-08 * T2 - 1.532250000000000e-12 * T3 +
                 5.873499999999999e-17 * T4;
    // species 4: O
    species[4] = +2.915000000000000e+04 * invT - 2.378200000000000e+00 -
                 2.542100000000000e+00 * logT + 1.377550000000000e-05 * T +
                 5.171333333333333e-10 * T2 - 3.792583333333333e-13 * T3 +
                 2.184050000000000e-17 * T4;
  }
}

// compute the a/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
helmholtz(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = -1.044000000000000e+03 * invT - 1.748200000000000e+00 -
                 3.614600000000000e+00 * logT + 9.299000000000000e-04 * T -
                 1.180233333333333e-06 * T2 + 5.672500000000000e-10 * T3 -
                 1.081400000000000e-13 * T4;
    // species 1: N2
    species[1] = -1.043000000000000e+03 * invT + 3.167999999999997e-01 -
                 3.674800000000000e+00 * logT + 6.040500000000000e-04 * T -
                 3.873333333333333e-07 * T2 + 5.268166666666667e-11 * T3 +
                 1.128850000000000e-14 * T4;
    // species 2: NO
    species[2] = +9.764000000000000e+03 * invT - 2.561000000000000e+00 -
                 3.588700000000000e+00 * logT + 6.239500000000000e-04 * T -
                 6.631000000000001e-07 * T2 + 2.387583333333333e-10 * T3 -
                 3.150750000000000e-14 * T4;
    // species 3: N
    species[3] = +5.613000000000000e+04 * invT - 2.664500000000000e+00 -
                 2.503100000000000e+00 * logT + 1.090000000000000e-05 * T -
                 9.034166666666666e-09 * T2 + 4.706333333333333e-12 * T3 -
                 1.049950000000000e-15 * T4;
    // species 4: O
    species[4] = +2.915000000000000e+04 * invT - 1.679100000000000e+00 -
                 2.823600000000000e+00 * logT + 4.473900000000000e-04 * T -
                 1.384333333333333e-07 * T2 + 1.403083333333333e-11 * T3 +
                 3.660250000000000e-15 * T4;
  } else {
    // species 0: O2
    species[0] = -1.044000000000000e+03 * invT - 1.240400000000000e+00 -
                 3.594900000000000e+00 * logT - 3.760650000000000e-04 * T +
                 3.122000000000000e-08 * T2 - 2.326083333333333e-12 * T3 +
                 7.887000000000000e-17 * T4;
    // species 1: N2
    species[1] = -1.043000000000000e+03 * invT - 2.153600000000000e+00 -
                 3.212500000000000e+00 * logT - 5.068500000000000e-04 * T +
                 5.077833333333333e-08 * T2 - 3.424250000000000e-12 * T3 +
                 1.008500000000000e-16 * T4;
    // species 2: NO
    species[2] = +9.764000000000000e+03 * invT - 4.482000000000000e+00 -
                 3.204700000000000e+00 * logT - 6.352500000000000e-04 * T +
                 7.767166666666667e-08 * T2 - 6.250583333333333e-12 * T3 +
                 2.115700000000000e-16 * T4;
    // species 3: N
    species[3] = +5.613000000000000e+04 * invT - 2.779800000000000e+00 -
                 2.482000000000000e+00 * logT - 3.462900000000000e-05 * T +
                 1.051083333333333e-08 * T2 - 1.532250000000000e-12 * T3 +
                 5.873499999999999e-17 * T4;
    // species 4: O
    species[4] = +2.915000000000000e+04 * invT - 3.378200000000000e+00 -
                 2.542100000000000e+00 * logT + 1.377550000000000e-05 * T +
                 5.171333333333333e-10 * T2 - 3.792583333333333e-13 * T3 +
                 2.184050000000000e-17 * T4;
  }
}

// compute the e/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesInternalEnergy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = +2.614600000000000e+00 - 9.299000000000000e-04 * T +
                 2.360466666666667e-06 * T2 - 1.701750000000000e-09 * T3 +
                 4.325600000000000e-13 * T4 - 1.044000000000000e+03 * invT;
    // species 1: N2
    species[1] = +2.674800000000000e+00 - 6.040500000000000e-04 * T +
                 7.746666666666667e-07 * T2 - 1.580450000000000e-10 * T3 -
                 4.515400000000000e-14 * T4 - 1.043000000000000e+03 * invT;
    // species 2: NO
    species[2] = +2.588700000000000e+00 - 6.239500000000000e-04 * T +
                 1.326200000000000e-06 * T2 - 7.162750000000000e-10 * T3 +
                 1.260300000000000e-13 * T4 + 9.764000000000000e+03 * invT;
    // species 3: N
    species[3] = +1.503100000000000e+00 - 1.090000000000000e-05 * T +
                 1.806833333333333e-08 * T2 - 1.411900000000000e-11 * T3 +
                 4.199800000000000e-15 * T4 + 5.613000000000000e+04 * invT;
    // species 4: O
    species[4] = +1.823600000000000e+00 - 4.473900000000000e-04 * T +
                 2.768666666666667e-07 * T2 - 4.209250000000000e-11 * T3 -
                 1.464100000000000e-14 * T4 + 2.915000000000000e+04 * invT;
  } else {
    // species 0: O2
    species[0] = +2.594900000000000e+00 + 3.760650000000000e-04 * T -
                 6.244000000000001e-08 * T2 + 6.978250000000000e-12 * T3 -
                 3.154800000000000e-16 * T4 - 1.044000000000000e+03 * invT;
    // species 1: N2
    species[1] = +2.212500000000000e+00 + 5.068500000000000e-04 * T -
                 1.015566666666667e-07 * T2 + 1.027275000000000e-11 * T3 -
                 4.034000000000000e-16 * T4 - 1.043000000000000e+03 * invT;
    // species 2: NO
    species[2] = +2.204700000000000e+00 + 6.352500000000000e-04 * T -
                 1.553433333333333e-07 * T2 + 1.875175000000000e-11 * T3 -
                 8.462800000000000e-16 * T4 + 9.764000000000000e+03 * invT;
    // species 3: N
    species[3] = +1.482000000000000e+00 + 3.462900000000000e-05 * T -
                 2.102166666666667e-08 * T2 + 4.596750000000000e-12 * T3 -
                 2.349400000000000e-16 * T4 + 5.613000000000000e+04 * invT;
    // species 4: O
    species[4] = +1.542100000000000e+00 - 1.377550000000000e-05 * T -
                 1.034266666666667e-09 * T2 + 1.137775000000000e-12 * T3 -
                 8.736200000000000e-17 * T4 + 2.915000000000000e+04 * invT;
  }
}

// compute the h/(RT) at the given temperature (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEnthalpy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = +3.614600000000000e+00 - 9.299000000000000e-04 * T +
                 2.360466666666667e-06 * T2 - 1.701750000000000e-09 * T3 +
                 4.325600000000000e-13 * T4 - 1.044000000000000e+03 * invT;
    // species 1: N2
    species[1] = +3.674800000000000e+00 - 6.040500000000000e-04 * T +
                 7.746666666666667e-07 * T2 - 1.580450000000000e-10 * T3 -
                 4.515400000000000e-14 * T4 - 1.043000000000000e+03 * invT;
    // species 2: NO
    species[2] = +3.588700000000000e+00 - 6.239500000000000e-04 * T +
                 1.326200000000000e-06 * T2 - 7.162750000000000e-10 * T3 +
                 1.260300000000000e-13 * T4 + 9.764000000000000e+03 * invT;
    // species 3: N
    species[3] = +2.503100000000000e+00 - 1.090000000000000e-05 * T +
                 1.806833333333333e-08 * T2 - 1.411900000000000e-11 * T3 +
                 4.199800000000000e-15 * T4 + 5.613000000000000e+04 * invT;
    // species 4: O
    species[4] = +2.823600000000000e+00 - 4.473900000000000e-04 * T +
                 2.768666666666667e-07 * T2 - 4.209250000000000e-11 * T3 -
                 1.464100000000000e-14 * T4 + 2.915000000000000e+04 * invT;
  } else {
    // species 0: O2
    species[0] = +3.594900000000000e+00 + 3.760650000000000e-04 * T -
                 6.244000000000001e-08 * T2 + 6.978250000000000e-12 * T3 -
                 3.154800000000000e-16 * T4 - 1.044000000000000e+03 * invT;
    // species 1: N2
    species[1] = +3.212500000000000e+00 + 5.068500000000000e-04 * T -
                 1.015566666666667e-07 * T2 + 1.027275000000000e-11 * T3 -
                 4.034000000000000e-16 * T4 - 1.043000000000000e+03 * invT;
    // species 2: NO
    species[2] = +3.204700000000000e+00 + 6.352500000000000e-04 * T -
                 1.553433333333333e-07 * T2 + 1.875175000000000e-11 * T3 -
                 8.462800000000000e-16 * T4 + 9.764000000000000e+03 * invT;
    // species 3: N
    species[3] = +2.482000000000000e+00 + 3.462900000000000e-05 * T -
                 2.102166666666667e-08 * T2 + 4.596750000000000e-12 * T3 -
                 2.349400000000000e-16 * T4 + 5.613000000000000e+04 * invT;
    // species 4: O
    species[4] = +2.542100000000000e+00 - 1.377550000000000e-05 * T -
                 1.034266666666667e-09 * T2 + 1.137775000000000e-12 * T3 -
                 8.736200000000000e-17 * T4 + 2.915000000000000e+04 * invT;
  }
}

// compute the S/R at the given temperature (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEntropy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real logT = log(T);

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = +3.614600000000000e+00 * logT - 1.859800000000000e-03 * T +
                 3.540700000000000e-06 * T2 - 2.269000000000000e-09 * T3 +
                 5.407000000000000e-13 * T4 + 4.362800000000000e+00;
    // species 1: N2
    species[1] = +3.674800000000000e+00 * logT - 1.208100000000000e-03 * T +
                 1.162000000000000e-06 * T2 - 2.107266666666667e-10 * T3 -
                 5.644250000000000e-14 * T4 + 2.358000000000000e+00;
    // species 2: NO
    species[2] = +3.588700000000000e+00 * logT - 1.247900000000000e-03 * T +
                 1.989300000000000e-06 * T2 - 9.550333333333333e-10 * T3 +
                 1.575375000000000e-13 * T4 + 5.149700000000000e+00;
    // species 3: N
    species[3] = +2.503100000000000e+00 * logT - 2.180000000000000e-05 * T +
                 2.710250000000000e-08 * T2 - 1.882533333333333e-11 * T3 +
                 5.249750000000000e-15 * T4 + 4.167600000000000e+00;
    // species 4: O
    species[4] = +2.823600000000000e+00 * logT - 8.947800000000000e-04 * T +
                 4.153000000000000e-07 * T2 - 5.612333333333333e-11 * T3 -
                 1.830125000000000e-14 * T4 + 3.502700000000000e+00;
  } else {
    // species 0: O2
    species[0] = +3.594900000000000e+00 * logT + 7.521300000000001e-04 * T -
                 9.366000000000001e-08 * T2 + 9.304333333333333e-12 * T3 -
                 3.943500000000000e-16 * T4 + 3.835300000000000e+00;
    // species 1: N2
    species[1] = +3.212500000000000e+00 * logT + 1.013700000000000e-03 * T -
                 1.523350000000000e-07 * T2 + 1.369700000000000e-11 * T3 -
                 5.042500000000000e-16 * T4 + 4.366100000000000e+00;
    // species 2: NO
    species[2] = +3.204700000000000e+00 * logT + 1.270500000000000e-03 * T -
                 2.330150000000000e-07 * T2 + 2.500233333333333e-11 * T3 -
                 1.057850000000000e-15 * T4 + 6.686700000000000e+00;
    // species 3: N
    species[3] = +2.482000000000000e+00 * logT + 6.925800000000001e-05 * T -
                 3.153250000000000e-08 * T2 + 6.129000000000000e-12 * T3 -
                 2.936750000000000e-16 * T4 + 4.261800000000000e+00;
    // species 4: O
    species[4] = +2.542100000000000e+00 * logT - 2.755100000000000e-05 * T -
                 1.551400000000000e-09 * T2 + 1.517033333333333e-12 * T3 -
                 1.092025000000000e-16 * T4 + 4.920300000000000e+00;
  }
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
dcvpRdT(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = -1.859800000000000e-03 + 1.416280000000000e-05 * T -
                 2.042100000000000e-08 * T2 + 8.651200000000000e-12 * T3;
    // species 1: N2
    species[1] = -1.208100000000000e-03 + 4.648000000000000e-06 * T -
                 1.896540000000000e-09 * T2 - 9.030800000000000e-13 * T3;
    // species 2: NO
    species[2] = -1.247900000000000e-03 + 7.957200000000001e-06 * T -
                 8.595300000000000e-09 * T2 + 2.520600000000000e-12 * T3;
    // species 3: N
    species[3] = -2.180000000000000e-05 + 1.084100000000000e-07 * T -
                 1.694280000000000e-10 * T2 + 8.399600000000000e-14 * T3;
    // species 4: O
    species[4] = -8.947800000000000e-04 + 1.661200000000000e-06 * T -
                 5.051100000000000e-10 * T2 - 2.928200000000000e-13 * T3;
  } else {
    // species 0: O2
    species[0] = +7.521300000000001e-04 - 3.746400000000000e-07 * T +
                 8.373900000000000e-11 * T2 - 6.309600000000000e-15 * T3;
    // species 1: N2
    species[1] = +1.013700000000000e-03 - 6.093400000000000e-07 * T +
                 1.232730000000000e-10 * T2 - 8.067999999999999e-15 * T3;
    // species 2: NO
    species[2] = +1.270500000000000e-03 - 9.320600000000000e-07 * T +
                 2.250210000000000e-10 * T2 - 1.692560000000000e-14 * T3;
    // species 3: N
    species[3] = +6.925800000000001e-05 - 1.261300000000000e-07 * T +
                 5.516100000000000e-11 * T2 - 4.698800000000000e-15 * T3;
    // species 4: O
    species[4] = -2.755100000000000e-05 - 6.205600000000000e-09 * T +
                 1.365330000000000e-11 * T2 - 1.747240000000000e-15 * T3;
  }
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
  amrex::Real result = 0;
  amrex::Real cpor[5]; //  temporary storage
  cp_R(cpor, T);

  // perform dot product
  for (int id = 0; id < 5; ++id) {
    result += x[id] * cpor[id];
  }

  cpbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CP (Eq. 34)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cpbs)
{
  amrex::Real result = 0.0;

  // compute Cp/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    result += y[0] *
              (+3.614600000000000e+00 - 1.859800000000000e-03 * T +
               7.081400000000000e-06 * T2 - 6.807000000000000e-09 * T3 +
               2.162800000000000e-12 * T4) *
              0.0312511719189470;
    // species 1: N2
    result += y[1] *
              (+3.674800000000000e+00 - 1.208100000000000e-03 * T +
               2.324000000000000e-06 * T2 - 6.321800000000000e-10 * T3 -
               2.257700000000000e-13 * T4) *
              0.0356972020533031;
    // species 2: NO
    result += y[2] *
              (+3.588700000000000e+00 - 1.247900000000000e-03 * T +
               3.978600000000000e-06 * T2 - 2.865100000000000e-09 * T3 +
               6.301500000000000e-13 * T4) *
              0.0333265569334235;
    // species 3: N
    result += y[3] *
              (+2.503100000000000e+00 - 2.180000000000000e-05 * T +
               5.420500000000000e-08 * T2 - 5.647600000000000e-11 * T3 +
               2.099900000000000e-14 * T4) *
              0.0713944041066061;
    // species 4: O
    result += y[4] *
              (+2.823600000000000e+00 - 8.947800000000000e-04 * T +
               8.306000000000000e-07 * T2 - 1.683700000000000e-10 * T3 -
               7.320500000000000e-14 * T4) *
              0.0625023438378939;
  } else {
    // species 0: O2
    result += y[0] *
              (+3.594900000000000e+00 + 7.521300000000001e-04 * T -
               1.873200000000000e-07 * T2 + 2.791300000000000e-11 * T3 -
               1.577400000000000e-15 * T4) *
              0.0312511719189470;
    // species 1: N2
    result += y[1] *
              (+3.212500000000000e+00 + 1.013700000000000e-03 * T -
               3.046700000000000e-07 * T2 + 4.109100000000000e-11 * T3 -
               2.017000000000000e-15 * T4) *
              0.0356972020533031;
    // species 2: NO
    result += y[2] *
              (+3.204700000000000e+00 + 1.270500000000000e-03 * T -
               4.660300000000000e-07 * T2 + 7.500699999999999e-11 * T3 -
               4.231400000000000e-15 * T4) *
              0.0333265569334235;
    // species 3: N
    result += y[3] *
              (+2.482000000000000e+00 + 6.925800000000001e-05 * T -
               6.306500000000000e-08 * T2 + 1.838700000000000e-11 * T3 -
               1.174700000000000e-15 * T4) *
              0.0713944041066061;
    // species 4: O
    result += y[4] *
              (+2.542100000000000e+00 - 2.755100000000000e-05 * T -
               3.102800000000000e-09 * T2 + 4.551100000000000e-12 * T3 -
               4.368100000000000e-16 * T4) *
              0.0625023438378939;
  }

  cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
  amrex::Real result = 0;
  amrex::Real cvor[5]; //  temporary storage
  cv_R(cvor, T);

  // perform dot product
  for (int id = 0; id < 5; ++id) {
    result += x[id] * cvor[id];
  }

  cvbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 36)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cvbs)
{
  amrex::Real result = 0.0;
  // compute Cv/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    result += y[0] *
              (+2.614600000000000e+00 - 1.859800000000000e-03 * T +
               7.081400000000000e-06 * T2 - 6.807000000000000e-09 * T3 +
               2.162800000000000e-12 * T4) *
              0.0312511719189470;
    // species 1: N2
    result += y[1] *
              (+2.674800000000000e+00 - 1.208100000000000e-03 * T +
               2.324000000000000e-06 * T2 - 6.321800000000000e-10 * T3 -
               2.257700000000000e-13 * T4) *
              0.0356972020533031;
    // species 2: NO
    result += y[2] *
              (+2.588700000000000e+00 - 1.247900000000000e-03 * T +
               3.978600000000000e-06 * T2 - 2.865100000000000e-09 * T3 +
               6.301500000000000e-13 * T4) *
              0.0333265569334235;
    // species 3: N
    result += y[3] *
              (+1.503100000000000e+00 - 2.180000000000000e-05 * T +
               5.420500000000000e-08 * T2 - 5.647600000000000e-11 * T3 +
               2.099900000000000e-14 * T4) *
              0.0713944041066061;
    // species 4: O
    result += y[4] *
              (+1.823600000000000e+00 - 8.947800000000000e-04 * T +
               8.306000000000000e-07 * T2 - 1.683700000000000e-10 * T3 -
               7.320500000000000e-14 * T4) *
              0.0625023438378939;
  } else {
    // species 0: O2
    result += y[0] *
              (+2.594900000000000e+00 + 7.521300000000001e-04 * T -
               1.873200000000000e-07 * T2 + 2.791300000000000e-11 * T3 -
               1.577400000000000e-15 * T4) *
              0.0312511719189470;
    // species 1: N2
    result += y[1] *
              (+2.212500000000000e+00 + 1.013700000000000e-03 * T -
               3.046700000000000e-07 * T2 + 4.109100000000000e-11 * T3 -
               2.017000000000000e-15 * T4) *
              0.0356972020533031;
    // species 2: NO
    result += y[2] *
              (+2.204700000000000e+00 + 1.270500000000000e-03 * T -
               4.660300000000000e-07 * T2 + 7.500699999999999e-11 * T3 -
               4.231400000000000e-15 * T4) *
              0.0333265569334235;
    // species 3: N
    result += y[3] *
              (+1.482000000000000e+00 + 6.925800000000001e-05 * T -
               6.306500000000000e-08 * T2 + 1.838700000000000e-11 * T3 -
               1.174700000000000e-15 * T4) *
              0.0713944041066061;
    // species 4: O
    result += y[4] *
              (+1.542100000000000e+00 - 2.755100000000000e-05 * T -
               3.102800000000000e-09 * T2 + 4.551100000000000e-12 * T3 -
               4.368100000000000e-16 * T4) *
              0.0625023438378939;
  }

  cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
  amrex::Real result = 0;
  amrex::Real hml[5];                        //  temporary storage
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // perform dot product
  for (int id = 0; id < 5; ++id) {
    result += x[id] * hml[id];
  }

  hbml = result * RT;
}

// Returns mean enthalpy of mixture in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& hbms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    result += y[0] *
              (+3.614600000000000e+00 - 9.299000000000000e-04 * T +
               2.360466666666667e-06 * T2 - 1.701750000000000e-09 * T3 +
               4.325600000000000e-13 * T4 - 1.044000000000000e+03 * invT) *
              0.0312511719189470;
    // species 1: N2
    result += y[1] *
              (+3.674800000000000e+00 - 6.040500000000000e-04 * T +
               7.746666666666667e-07 * T2 - 1.580450000000000e-10 * T3 -
               4.515400000000000e-14 * T4 - 1.043000000000000e+03 * invT) *
              0.0356972020533031;
    // species 2: NO
    result += y[2] *
              (+3.588700000000000e+00 - 6.239500000000000e-04 * T +
               1.326200000000000e-06 * T2 - 7.162750000000000e-10 * T3 +
               1.260300000000000e-13 * T4 + 9.764000000000000e+03 * invT) *
              0.0333265569334235;
    // species 3: N
    result += y[3] *
              (+2.503100000000000e+00 - 1.090000000000000e-05 * T +
               1.806833333333333e-08 * T2 - 1.411900000000000e-11 * T3 +
               4.199800000000000e-15 * T4 + 5.613000000000000e+04 * invT) *
              0.0713944041066061;
    // species 4: O
    result += y[4] *
              (+2.823600000000000e+00 - 4.473900000000000e-04 * T +
               2.768666666666667e-07 * T2 - 4.209250000000000e-11 * T3 -
               1.464100000000000e-14 * T4 + 2.915000000000000e+04 * invT) *
              0.0625023438378939;
  } else {
    // species 0: O2
    result += y[0] *
              (+3.594900000000000e+00 + 3.760650000000000e-04 * T -
               6.244000000000001e-08 * T2 + 6.978250000000000e-12 * T3 -
               3.154800000000000e-16 * T4 - 1.044000000000000e+03 * invT) *
              0.0312511719189470;
    // species 1: N2
    result += y[1] *
              (+3.212500000000000e+00 + 5.068500000000000e-04 * T -
               1.015566666666667e-07 * T2 + 1.027275000000000e-11 * T3 -
               4.034000000000000e-16 * T4 - 1.043000000000000e+03 * invT) *
              0.0356972020533031;
    // species 2: NO
    result += y[2] *
              (+3.204700000000000e+00 + 6.352500000000000e-04 * T -
               1.553433333333333e-07 * T2 + 1.875175000000000e-11 * T3 -
               8.462800000000000e-16 * T4 + 9.764000000000000e+03 * invT) *
              0.0333265569334235;
    // species 3: N
    result += y[3] *
              (+2.482000000000000e+00 + 3.462900000000000e-05 * T -
               2.102166666666667e-08 * T2 + 4.596750000000000e-12 * T3 -
               2.349400000000000e-16 * T4 + 5.613000000000000e+04 * invT) *
              0.0713944041066061;
    // species 4: O
    result += y[4] *
              (+2.542100000000000e+00 - 1.377550000000000e-05 * T -
               1.034266666666667e-09 * T2 + 1.137775000000000e-12 * T3 -
               8.736200000000000e-17 * T4 + 2.915000000000000e+04 * invT) *
              0.0625023438378939;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
  amrex::Real result = 0;
  amrex::Real uml[5];                        //  temporary energy array
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // perform dot product
  for (int id = 0; id < 5; ++id) {
    result += x[id] * uml[id];
  }

  ubml = result * RT;
}

// get mean internal energy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& ubms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    result += y[0] *
              (+2.614600000000000e+00 - 9.299000000000000e-04 * T +
               2.360466666666667e-06 * T2 - 1.701750000000000e-09 * T3 +
               4.325600000000000e-13 * T4 - 1.044000000000000e+03 * invT) *
              0.0312511719189470;
    // species 1: N2
    result += y[1] *
              (+2.674800000000000e+00 - 6.040500000000000e-04 * T +
               7.746666666666667e-07 * T2 - 1.580450000000000e-10 * T3 -
               4.515400000000000e-14 * T4 - 1.043000000000000e+03 * invT) *
              0.0356972020533031;
    // species 2: NO
    result += y[2] *
              (+2.588700000000000e+00 - 6.239500000000000e-04 * T +
               1.326200000000000e-06 * T2 - 7.162750000000000e-10 * T3 +
               1.260300000000000e-13 * T4 + 9.764000000000000e+03 * invT) *
              0.0333265569334235;
    // species 3: N
    result += y[3] *
              (+1.503100000000000e+00 - 1.090000000000000e-05 * T +
               1.806833333333333e-08 * T2 - 1.411900000000000e-11 * T3 +
               4.199800000000000e-15 * T4 + 5.613000000000000e+04 * invT) *
              0.0713944041066061;
    // species 4: O
    result += y[4] *
              (+1.823600000000000e+00 - 4.473900000000000e-04 * T +
               2.768666666666667e-07 * T2 - 4.209250000000000e-11 * T3 -
               1.464100000000000e-14 * T4 + 2.915000000000000e+04 * invT) *
              0.0625023438378939;
  } else {
    // species 0: O2
    result += y[0] *
              (+2.594900000000000e+00 + 3.760650000000000e-04 * T -
               6.244000000000001e-08 * T2 + 6.978250000000000e-12 * T3 -
               3.154800000000000e-16 * T4 - 1.044000000000000e+03 * invT) *
              0.0312511719189470;
    // species 1: N2
    result += y[1] *
              (+2.212500000000000e+00 + 5.068500000000000e-04 * T -
               1.015566666666667e-07 * T2 + 1.027275000000000e-11 * T3 -
               4.034000000000000e-16 * T4 - 1.043000000000000e+03 * invT) *
              0.0356972020533031;
    // species 2: NO
    result += y[2] *
              (+2.204700000000000e+00 + 6.352500000000000e-04 * T -
               1.553433333333333e-07 * T2 + 1.875175000000000e-11 * T3 -
               8.462800000000000e-16 * T4 + 9.764000000000000e+03 * invT) *
              0.0333265569334235;
    // species 3: N
    result += y[3] *
              (+1.482000000000000e+00 + 3.462900000000000e-05 * T -
               2.102166666666667e-08 * T2 + 4.596750000000000e-12 * T3 -
               2.349400000000000e-16 * T4 + 5.613000000000000e+04 * invT) *
              0.0713944041066061;
    // species 4: O
    result += y[4] *
              (+1.542100000000000e+00 - 1.377550000000000e-05 * T -
               1.034266666666667e-09 * T2 + 1.137775000000000e-12 * T3 -
               8.736200000000000e-17 * T4 + 2.915000000000000e+04 * invT) *
              0.0625023438378939;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  ubms = result * RT;
}

// get mixture entropy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBML(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& sbml)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[5]; //  temporary storage
  speciesEntropy(sor, T);

  // Compute Eq 42
  for (int id = 0; id < 5; ++id) {
    result += x[id] * (sor[id] - log((x[id] + 1e-100)) - logPratio);
  }

  sbml = result * 8.31446261815324e+07;
}

// get mixture entropy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBMS(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& sbms)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[5];  //  temporary storage
  amrex::Real x[5];    //  need a ytx conversion
  amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 5; i++) {
    YOW += y[i] * imw(i);
  }
  // Now compute y to x conversion
  x[0] = y[0] / (3.1998799999999999e+01 * YOW);
  x[1] = y[1] / (2.8013400000000001e+01 * YOW);
  x[2] = y[2] / (3.0006100000000000e+01 * YOW);
  x[3] = y[3] / (1.4006700000000000e+01 * YOW);
  x[4] = y[4] / (1.5999400000000000e+01 * YOW);
  speciesEntropy(sor, T);
  // Perform computation in Eq 42 and 43
  for (int i = 0; i < 5; i++) {
    result += x[i] * (sor[i] - log((x[i] + 1e-100)) - logPratio);
  }
  // Scale by R/W
  sbms = result * 8.31446261815324e+07 * YOW;
}

//  get temperature given internal energy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_EY(
  const amrex::Real e, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 200.0;  // max lower bound for thermo def
  amrex::Real tmax = 6000.0; // min upper bound for thermo def
  amrex::Real e1, emin, emax, cv, t1, dt;
  CKUBMS(tmin, y, emin);
  CKUBMS(tmax, y, emax);
  if (e < emin) {
    // Linear Extrapolation below tmin
    CKCVBS(tmin, y, cv);
    t = tmin - (emin - e) / cv;
    ierr = 1;
    return;
  }
  if (e > emax) {
    // Linear Extrapolation above tmax
    CKCVBS(tmax, y, cv);
    t = tmax - (emax - e) / cv;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (emax - emin) * (e - emin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKUBMS(t1, y, e1);
    CKCVBS(t1, y, cv);
    dt = (e - e1) / cv;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

//  get temperature given enthalpy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_HY(
  const amrex::Real h, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 200.0;  // max lower bound for thermo def
  amrex::Real tmax = 6000.0; // min upper bound for thermo def
  amrex::Real h1, hmin, hmax, cp, t1, dt;
  CKHBMS(tmin, y, hmin);
  CKHBMS(tmax, y, hmax);
  if (h < hmin) {
    // Linear Extrapolation below tmin
    CKCPBS(tmin, y, cp);
    t = tmin - (hmin - h) / cp;
    ierr = 1;
    return;
  }
  if (h > hmax) {
    // Linear Extrapolation above tmax
    CKCPBS(tmax, y, cp);
    t = tmax - (hmax - h) / cp;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (hmax - hmin) * (h - hmin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKHBMS(t1, y, h1);
    CKCPBS(t1, y, cp);
    dt = (h - h1) / cp;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

// Compute P = rhoRT/W(x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPX(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& P)
{
  amrex::Real XW = 0;                      //  To hold mean molecular wt
  XW += x[0] * 3.1998799999999999e+01;     // O2
  XW += x[1] * 2.8013400000000001e+01;     // N2
  XW += x[2] * 3.0006100000000000e+01;     // NO
  XW += x[3] * 1.4006700000000000e+01;     // N
  XW += x[4] * 1.5999400000000000e+01;     // O
  P = rho * 8.31446261815324e+07 * T / XW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPY(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& P)
{
  amrex::Real YOW = 0; //  for computing mean MW

  for (int i = 0; i < 5; i++) {
    YOW += y[i] * imw(i);
  }
  P = rho * 8.31446261815324e+07 * T * YOW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(c)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPC(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& P)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 3.1998799999999999e+01; // O2
  W += c[1] * 2.8013400000000001e+01; // N2
  W += c[2] * 3.0006100000000000e+01; // NO
  W += c[3] * 1.4006700000000000e+01; // N
  W += c[4] * 1.5999400000000000e+01; // O

  for (int id = 0; id < 5; ++id) {
    sumC += c[id];
  }
  P = rho * 8.31446261815324e+07 * T * sumC / W; // P = rho*R*T/W
}

// Compute rho = PW(x)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOX(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& rho)
{
  amrex::Real XW = 0;                        //  To hold mean molecular wt
  XW += x[0] * 3.1998799999999999e+01;       // O2
  XW += x[1] * 2.8013400000000001e+01;       // N2
  XW += x[2] * 3.0006100000000000e+01;       // NO
  XW += x[3] * 1.4006700000000000e+01;       // N
  XW += x[4] * 1.5999400000000000e+01;       // O
  rho = P * XW / (8.31446261815324e+07 * T); // rho = P*W/(R*T)
}

// Compute rho = P*W(y)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOY(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& rho)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 5; i++) {
    YOW += y[i] * imw(i);
  }

  rho = P / (8.31446261815324e+07 * T * YOW); // rho = P*W/(R*T)
}

// Compute rho = P*W(c)/(R*T)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOC(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& rho)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 3.1998799999999999e+01; // O2
  W += c[1] * 2.8013400000000001e+01; // N2
  W += c[2] * 3.0006100000000000e+01; // NO
  W += c[3] * 1.4006700000000000e+01; // N
  W += c[4] * 1.5999400000000000e+01; // O

  for (int id = 0; id < 5; ++id) {
    sumC += c[id];
  }
  rho = P * W / (sumC * T * 8.31446261815324e+07); // rho = PW/(R*T)
}

// get molecular weight for all species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWT(amrex::Real wt[])
{
  get_mw(wt);
}

// given y[species]: mass fractions
// s mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWY(const amrex::Real y[], amrex::Real& wtm)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 5; i++) {
    YOW += y[i] * imw(i);
  }

  wtm = 1.0 / YOW;
}

// given x[species]: mole fractions
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWX(const amrex::Real x[], amrex::Real& wtm)
{
  amrex::Real XW = 0;                  //  see Eq 4 in CK Manual
  XW += x[0] * 3.1998799999999999e+01; // O2
  XW += x[1] * 2.8013400000000001e+01; // N2
  XW += x[2] * 3.0006100000000000e+01; // NO
  XW += x[3] * 1.4006700000000000e+01; // N
  XW += x[4] * 1.5999400000000000e+01; // O
  wtm = XW;
}

// given c[species]: molar concentration
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWC(const amrex::Real c[], amrex::Real& wtm)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 3.1998799999999999e+01; // O2
  W += c[1] * 2.8013400000000001e+01; // N2
  W += c[2] * 3.0006100000000000e+01; // NO
  W += c[3] * 1.4006700000000000e+01; // N
  W += c[4] * 1.5999400000000000e+01; // O

  for (int id = 0; id < 5; ++id) {
    sumC += c[id];
  }
  //  CK provides no guard against division by zero
  wtm = W / sumC;
}

// get Cp/R as a function of T
// for all species (Eq 19)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPOR(const amrex::Real T, amrex::Real cpor[])
{
  cp_R(cpor, T);
}

// get H/RT as a function of T
// for all species (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHORT(const amrex::Real T, amrex::Real hort[])
{
  speciesEnthalpy(hort, T);
}

// get S/R as a function of T
// for all species (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSOR(const amrex::Real T, amrex::Real sor[])
{
  speciesEntropy(sor, T);
}

// convert y[species] (mass fracs) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTX(const amrex::Real y[], amrex::Real x[])
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 5; i++) {
    YOW += y[i] * imw(i);
  }

  amrex::Real YOWINV = 1.0 / YOW;

  for (int i = 0; i < 5; i++) {
    x[i] = y[i] * imw(i) * YOWINV;
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real c[])
{
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 5; i++) {
    c[i] = y[i] * imw(i);
  }
  for (int i = 0; i < 5; i++) {
    YOW += c[i];
  }

  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // Now compute conversion

  for (int i = 0; i < 5; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCR(
  const amrex::Real rho,
  amrex::Real /*T*/,
  const amrex::Real y[],
  amrex::Real c[])
{

  for (int i = 0; i < 5; i++) {
    c[i] = rho * y[i] * imw(i);
  }
}

// convert x[species] (mole fracs) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTY(const amrex::Real x[], amrex::Real y[])
{
  amrex::Real XW = 0; // See Eq 4, 9 in CK Manual
  // Compute mean molecular wt first
  XW += x[0] * 3.1998799999999999e+01; // O2
  XW += x[1] * 2.8013400000000001e+01; // N2
  XW += x[2] * 3.0006100000000000e+01; // NO
  XW += x[3] * 1.4006700000000000e+01; // N
  XW += x[4] * 1.5999400000000000e+01; // O
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 3.1998799999999999e+01 * XWinv;
  y[1] = x[1] * 2.8013400000000001e+01 * XWinv;
  y[2] = x[2] * 3.0006100000000000e+01 * XWinv;
  y[3] = x[3] * 1.4006700000000000e+01 * XWinv;
  y[4] = x[4] * 1.5999400000000000e+01 * XWinv;
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real PORT = P / (8.31446261815324e+07 * T); // P/RT

  // Compute conversion, see Eq 10
  for (int id = 0; id < 5; ++id) {
    c[id] = x[id] * PORT;
  }
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCR(
  const amrex::Real rho,
  const amrex::Real /*T*/,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 3.1998799999999999e+01; // O2
  XW += x[1] * 2.8013400000000001e+01; // N2
  XW += x[2] * 3.0006100000000000e+01; // NO
  XW += x[3] * 1.4006700000000000e+01; // N
  XW += x[4] * 1.5999400000000000e+01; // O
  ROW = rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 5; ++id) {
    c[id] = x[id] * ROW;
  }
}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTX(const amrex::Real c[], amrex::Real x[])
{
  amrex::Real sumC = 0;

  // compute sum of c
  for (int id = 0; id < 5; ++id) {
    sumC += c[id];
  }

  //  See Eq 13
  amrex::Real sumCinv = 1.0 / sumC;
  for (int id = 0; id < 5; ++id) {
    x[id] = c[id] * sumCinv;
  }
}

// convert c[species] (molar conc) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTY(const amrex::Real c[], amrex::Real y[])
{
  amrex::Real CW = 0; // See Eq 12 in CK Manual
  // compute denominator in eq 12 first
  CW += c[0] * 3.1998799999999999e+01; // O2
  CW += c[1] * 2.8013400000000001e+01; // N2
  CW += c[2] * 3.0006100000000000e+01; // NO
  CW += c[3] * 1.4006700000000000e+01; // N
  CW += c[4] * 1.5999400000000000e+01; // O
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 3.1998799999999999e+01 * CWinv;
  y[1] = c[1] * 2.8013400000000001e+01 * CWinv;
  y[2] = c[2] * 3.0006100000000000e+01 * CWinv;
  y[3] = c[3] * 1.4006700000000000e+01 * CWinv;
  y[4] = c[4] * 1.5999400000000000e+01 * CWinv;
}

// get specific heat at constant volume as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVML(const amrex::Real T, amrex::Real cvml[])
{
  cv_R(cvml, T);

  // convert to chemkin units
  for (int id = 0; id < 5; ++id) {
    cvml[id] *= 8.31446261815324e+07;
  }
}

// get specific heat at constant pressure as a
// function of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPML(const amrex::Real T, amrex::Real cpml[])
{
  cp_R(cpml, T);

  // convert to chemkin units
  for (int id = 0; id < 5; ++id) {
    cpml[id] *= 8.31446261815324e+07;
  }
}

// get internal energy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUML(const amrex::Real T, amrex::Real uml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // convert to chemkin units
  for (int id = 0; id < 5; ++id) {
    uml[id] *= RT;
  }
}

// get enthalpy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHML(const amrex::Real T, amrex::Real hml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // convert to chemkin units
  for (int id = 0; id < 5; ++id) {
    hml[id] *= RT;
  }
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSML(const amrex::Real T, amrex::Real sml[])
{
  speciesEntropy(sml, T);

  // convert to chemkin units
  for (int id = 0; id < 5; ++id) {
    sml[id] *= 8.31446261815324e+07;
  }
}

// Returns the specific heats at constant volume
// in mass units (Eq. 29)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVMS(const amrex::Real T, amrex::Real cvms[])
{
  cv_R(cvms, T);
  // multiply by R/molecularweight
  cvms[0] *= 2.598367006935648e+06; // O2
  cvms[1] *= 2.968030520448514e+06; // N2
  cvms[2] *= 2.770924118147057e+06; // NO
  cvms[3] *= 5.936061040897028e+06; // N
  cvms[4] *= 5.196734013871295e+06; // O
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
  cp_R(cpms, T);
  // multiply by R/molecularweight
  cpms[0] *= 2.598367006935648e+06; // O2
  cpms[1] *= 2.968030520448514e+06; // N2
  cpms[2] *= 2.770924118147057e+06; // NO
  cpms[3] *= 5.936061040897028e+06; // N
  cpms[4] *= 5.196734013871295e+06; // O
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUMS(const amrex::Real T, amrex::Real ums[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesInternalEnergy(ums, T);

  for (int i = 0; i < 5; i++) {
    ums[i] *= RT * imw(i);
  }
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHMS(const amrex::Real T, amrex::Real hms[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesEnthalpy(hms, T);

  for (int i = 0; i < 5; i++) {
    hms[i] *= RT * imw(i);
  }
}

// Returns the entropies in mass units (Eq 28.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSMS(const amrex::Real T, amrex::Real sms[])
{
  speciesEntropy(sms, T);
  // multiply by R/molecularweight
  sms[0] *= 2.598367006935648e+06; // O2
  sms[1] *= 2.968030520448514e+06; // N2
  sms[2] *= 2.770924118147057e+06; // NO
  sms[3] *= 5.936061040897028e+06; // N
  sms[4] *= 5.196734013871295e+06; // O
}

// GPU version of productionRate: no more use of thermo namespace vectors
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
comp_qfqr(
  amrex::Real* qf,
  amrex::Real* qr,
  const amrex::Real* sc,
  const amrex::Real* /*sc_qss*/,
  const amrex::Real T,
  const amrex::Real invT,
  const amrex::Real logT)
{

  // reaction 0: N2 + M <=> 2 N + M
  qf[0] = sc[1];
  qr[0] = (sc[3] * sc[3]);

  // reaction 1: O2 + M <=> 2 O + M
  qf[1] = sc[0];
  qr[1] = (sc[4] * sc[4]);

  // reaction 2: NO + M <=> N + O + M
  qf[2] = sc[2];
  qr[2] = sc[3] * sc[4];

  // reaction 3: N2 + O <=> N + NO
  qf[3] = sc[1] * sc[4];
  qr[3] = sc[2] * sc[3];

  // reaction 4: NO + O <=> N + O2
  qf[4] = sc[2] * sc[4];
  qr[4] = sc[0] * sc[3];

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 5; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[5];
  gibbs(g_RT, T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 * invT;
  amrex::Real refCinv = 1 / refC;

  // Evaluate the kfs
  amrex::Real k_f, Corr;

  // reaction 0:  N2 + M <=> 2 N + M
  k_f = 7e+15 * exp((-1.6) * logT - (113199.092189437) * invT);
  Corr = mixture + (3.2857) * sc[3] + (3.2857) * sc[4];
  qf[0] *= Corr * k_f;
  qr[0] *= Corr * k_f * exp(-(g_RT[1] - 2.000000 * g_RT[3])) * (refCinv);
  // reaction 1:  O2 + M <=> 2 O + M
  k_f = 2e+15 * exp((-1.5) * logT - (59499.3321314184) * invT);
  Corr = mixture + (4) * sc[3] + (4) * sc[4];
  qf[1] *= Corr * k_f;
  qr[1] *= Corr * k_f * exp(-(g_RT[0] - 2.000000 * g_RT[4])) * (refCinv);
  // reaction 2:  NO + M <=> N + O + M
  k_f = 5000000000 * exp(-(75499.6092373453) * invT);
  Corr = mixture + (21) * sc[3] + (21) * sc[4];
  qf[2] *= Corr * k_f;
  qr[2] *= Corr * k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[4])) * (refCinv);
  // reaction 3:  N2 + O <=> N + NO
  k_f = 640000000000 * exp((-1) * logT - (38399.4573342266) * invT);
  qf[3] *= k_f;
  qr[3] *= k_f * exp(-(g_RT[1] - g_RT[2] - g_RT[3] + g_RT[4]));
  // reaction 4:  NO + O <=> N + O2
  k_f = 8400000 * exp(-(19449.8273500182) * invT);
  qf[4] *= k_f;
  qr[4] *= k_f * exp(-(-g_RT[0] + g_RT[2] - g_RT[3] + g_RT[4]));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
productionRate(amrex::Real* wdot, const amrex::Real* sc, const amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  const amrex::Real refC = 101325 / 8.31446 * invT;
  const amrex::Real refCinv = 1 / refC;

  for (int i = 0; i < 5; ++i) {
    wdot[i] = 0.0;
  }

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 5; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[5];
  gibbs(g_RT, T);

  {
    // reaction 0:  N2 + M <=> 2 N + M
    const amrex::Real k_f =
      7e+15 * exp((-1.6) * logT - (113199.092189437) * invT);
    const amrex::Real Corr = mixture + (3.2857) * sc[3] + (3.2857) * sc[4];
    const amrex::Real qf = Corr * k_f * (sc[1]);
    const amrex::Real qr = Corr * k_f * exp(-(g_RT[1] - 2.000000 * g_RT[3])) *
                           (refCinv) * ((sc[3] * sc[3]));
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[3] += 2.000000 * qdot;
  }

  {
    // reaction 1:  O2 + M <=> 2 O + M
    const amrex::Real k_f =
      2e+15 * exp((-1.5) * logT - (59499.3321314184) * invT);
    const amrex::Real Corr = mixture + (4) * sc[3] + (4) * sc[4];
    const amrex::Real qf = Corr * k_f * (sc[0]);
    const amrex::Real qr = Corr * k_f * exp(-(g_RT[0] - 2.000000 * g_RT[4])) *
                           (refCinv) * ((sc[4] * sc[4]));
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[4] += 2.000000 * qdot;
  }

  {
    // reaction 2:  NO + M <=> N + O + M
    const amrex::Real k_f = 5000000000 * exp(-(75499.6092373453) * invT);
    const amrex::Real Corr = mixture + (21) * sc[3] + (21) * sc[4];
    const amrex::Real qf = Corr * k_f * (sc[2]);
    const amrex::Real qr = Corr * k_f * exp(-(g_RT[2] - g_RT[3] - g_RT[4])) *
                           (refCinv) * (sc[3] * sc[4]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[4] += qdot;
  }

  {
    // reaction 3:  N2 + O <=> N + NO
    const amrex::Real k_f =
      640000000000 * exp((-1) * logT - (38399.4573342266) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(g_RT[1] - g_RT[2] - g_RT[3] + g_RT[4])) * (sc[2] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[3] += qdot;
    wdot[4] -= qdot;
  }

  {
    // reaction 4:  NO + O <=> N + O2
    const amrex::Real k_f = 8400000 * exp(-(19449.8273500182) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[4]);
    const amrex::Real qr =
      k_f * exp(-(-g_RT[0] + g_RT[2] - g_RT[3] + g_RT[4])) * (sc[0] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[4] -= qdot;
  }
}

// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWC(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])
{

  // convert to SI
  for (int id = 0; id < 5; ++id) {
    C[id] *= 1.0e6;
  }

  // convert to chemkin units
  productionRate(wdot, C, T);

  // convert to chemkin units
  for (int id = 0; id < 5; ++id) {
    C[id] *= 1.0e-6;
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[5]; // temporary storage
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 5; i++) {
    YOW += y[i] * imw(i);
  }
  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // multiply by 1e6 so c goes to SI
  PWORT *= 1e6;
  // Now compute conversion (and go to SI)
  for (int i = 0; i < 5; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 5; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[5]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 5; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 5; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[5]; // temporary storage

  // See Eq 8 with an extra 1e6 so c goes to SI
  for (int i = 0; i < 5; i++) {
    c[i] = 1e6 * rho * y[i] * imw(i);
  }

  // call productionRate
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 5; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[5];   // temporary storage
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 3.1998799999999999e+01; // O2
  XW += x[1] * 2.8013400000000001e+01; // N2
  XW += x[2] * 3.0006100000000000e+01; // NO
  XW += x[3] * 1.4006700000000000e+01; // N
  XW += x[4] * 1.5999400000000000e+01; // O
  // Extra 1e6 factor to take c to SI
  ROW = 1e6 * rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 5; ++id) {
    c[id] = x[id] * ROW;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 5; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRG(int kcharge[])
{
  kcharge[0] = 0; // O2
  kcharge[1] = 0; // N2
  kcharge[2] = 0; // NO
  kcharge[3] = 0; // N
  kcharge[4] = 0; // O
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRGMASS(amrex::Real zk[])
{

  int kchrg[5];
  CKCHRG(kchrg);

  for (int id = 0; id < 5; ++id) {
    zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
  }
}

// compute an approx to the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian_precond(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int HP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 36; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[5];
  for (auto& val : wdot) {
    val = 0.0;
  }

  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 / T;
  amrex::Real refCinv = 1.0 / refC;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int k = 0; k < 5; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[5];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[5];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[5];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 0: N2 + M <=> 2 N + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (4.2857 - 1) * sc[3] + (4.2857 - 1) * sc[4];
  // forward
  phi_f = sc[1];
  k_f = 7e+15 * exp(-1.6 * logT - (113199.092189437) * invT);
  dlnkfdT = -1.6 * invT + (113199.092189437) * invT2;
  // reverse
  phi_r = (sc[3] * sc[3]);
  Kc = refC * exp(g_RT[1] - 2.000000 * g_RT[3]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1]) + (2.000000 * h_RT[3]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q;     // N2
  wdot[3] += 2 * q; // N
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor + k_f;
  dqdc[2] = q_nocor;
  dqdc[3] = 4.2857 * q_nocor - k_r * 2.000000 * sc[3];
  dqdc[4] = 4.2857 * q_nocor;
  for (int k = 0; k < 5; k++) {
    J[6 * k + 1] -= dqdc[k];
    J[6 * k + 3] += 2 * dqdc[k];
  }
  J[31] -= dqdT;     // dwdot[N2]/dT
  J[33] += 2 * dqdT; // dwdot[N]/dT

  // reaction 1: O2 + M <=> 2 O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (5 - 1) * sc[3] + (5 - 1) * sc[4];
  // forward
  phi_f = sc[0];
  k_f = 2e+15 * exp(-1.5 * logT - (59499.3321314184) * invT);
  dlnkfdT = -1.5 * invT + (59499.3321314184) * invT2;
  // reverse
  phi_r = (sc[4] * sc[4]);
  Kc = refC * exp(g_RT[0] - 2.000000 * g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0]) + (2.000000 * h_RT[4]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q;     // O2
  wdot[4] += 2 * q; // O
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor + k_f;
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor;
  dqdc[3] = 5 * q_nocor;
  dqdc[4] = 5 * q_nocor - k_r * 2.000000 * sc[4];
  for (int k = 0; k < 5; k++) {
    J[6 * k + 0] -= dqdc[k];
    J[6 * k + 4] += 2 * dqdc[k];
  }
  J[30] -= dqdT;     // dwdot[O2]/dT
  J[34] += 2 * dqdT; // dwdot[O]/dT

  // reaction 2: NO + M <=> N + O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (22 - 1) * sc[3] + (22 - 1) * sc[4];
  // forward
  phi_f = sc[2];
  k_f = 5000000000 * exp(-(75499.6092373453) * invT);
  dlnkfdT = (75499.6092373453) * invT2;
  // reverse
  phi_r = sc[3] * sc[4];
  Kc = refC * exp(g_RT[2] - g_RT[3] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2]) + (h_RT[3] + h_RT[4]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // NO
  wdot[3] += q; // N
  wdot[4] += q; // O
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor + k_f;
  dqdc[3] = 22 * q_nocor - k_r * sc[4];
  dqdc[4] = 22 * q_nocor - k_r * sc[3];
  for (int k = 0; k < 5; k++) {
    J[6 * k + 2] -= dqdc[k];
    J[6 * k + 3] += dqdc[k];
    J[6 * k + 4] += dqdc[k];
  }
  J[32] -= dqdT; // dwdot[NO]/dT
  J[33] += dqdT; // dwdot[N]/dT
  J[34] += dqdT; // dwdot[O]/dT

  // reaction 3: N2 + O <=> N + NO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[4];
  k_f = 640000000000 * exp(-1 * logT - (38399.4573342266) * invT);
  dlnkfdT = -1 * invT + (38399.4573342266) * invT2;
  // reverse
  phi_r = sc[2] * sc[3];
  Kc = exp(g_RT[1] - g_RT[2] - g_RT[3] + g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[4]) + (h_RT[2] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // N2
  wdot[2] += q; // NO
  wdot[3] += q; // N
  wdot[4] -= q; // O
  // d()/d[N2]
  dqdci = +k_f * sc[4];
  J[7] -= dqdci;  // dwdot[N2]/d[N2]
  J[8] += dqdci;  // dwdot[NO]/d[N2]
  J[9] += dqdci;  // dwdot[N]/d[N2]
  J[10] -= dqdci; // dwdot[O]/d[N2]
  // d()/d[NO]
  dqdci = -k_r * sc[3];
  J[13] -= dqdci; // dwdot[N2]/d[NO]
  J[14] += dqdci; // dwdot[NO]/d[NO]
  J[15] += dqdci; // dwdot[N]/d[NO]
  J[16] -= dqdci; // dwdot[O]/d[NO]
  // d()/d[N]
  dqdci = -k_r * sc[2];
  J[19] -= dqdci; // dwdot[N2]/d[N]
  J[20] += dqdci; // dwdot[NO]/d[N]
  J[21] += dqdci; // dwdot[N]/d[N]
  J[22] -= dqdci; // dwdot[O]/d[N]
  // d()/d[O]
  dqdci = +k_f * sc[1];
  J[25] -= dqdci; // dwdot[N2]/d[O]
  J[26] += dqdci; // dwdot[NO]/d[O]
  J[27] += dqdci; // dwdot[N]/d[O]
  J[28] -= dqdci; // dwdot[O]/d[O]
  // d()/dT
  J[31] -= dqdT; // dwdot[N2]/dT
  J[32] += dqdT; // dwdot[NO]/dT
  J[33] += dqdT; // dwdot[N]/dT
  J[34] -= dqdT; // dwdot[O]/dT

  // reaction 4: NO + O <=> N + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[4];
  k_f = 8400000 * exp(-(19449.8273500182) * invT);
  dlnkfdT = (19449.8273500182) * invT2;
  // reverse
  phi_r = sc[0] * sc[3];
  Kc = exp(-g_RT[0] + g_RT[2] - g_RT[3] + g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[4]) + (h_RT[0] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // O2
  wdot[2] -= q; // NO
  wdot[3] += q; // N
  wdot[4] -= q; // O
  // d()/d[O2]
  dqdci = -k_r * sc[3];
  J[0] += dqdci; // dwdot[O2]/d[O2]
  J[2] -= dqdci; // dwdot[NO]/d[O2]
  J[3] += dqdci; // dwdot[N]/d[O2]
  J[4] -= dqdci; // dwdot[O]/d[O2]
  // d()/d[NO]
  dqdci = +k_f * sc[4];
  J[12] += dqdci; // dwdot[O2]/d[NO]
  J[14] -= dqdci; // dwdot[NO]/d[NO]
  J[15] += dqdci; // dwdot[N]/d[NO]
  J[16] -= dqdci; // dwdot[O]/d[NO]
  // d()/d[N]
  dqdci = -k_r * sc[0];
  J[18] += dqdci; // dwdot[O2]/d[N]
  J[20] -= dqdci; // dwdot[NO]/d[N]
  J[21] += dqdci; // dwdot[N]/d[N]
  J[22] -= dqdci; // dwdot[O]/d[N]
  // d()/d[O]
  dqdci = +k_f * sc[2];
  J[24] += dqdci; // dwdot[O2]/d[O]
  J[26] -= dqdci; // dwdot[NO]/d[O]
  J[27] += dqdci; // dwdot[N]/d[O]
  J[28] -= dqdci; // dwdot[O]/d[O]
  // d()/dT
  J[30] += dqdT; // dwdot[O2]/dT
  J[32] -= dqdT; // dwdot[NO]/dT
  J[33] += dqdT; // dwdot[N]/dT
  J[34] -= dqdT; // dwdot[O]/dT

  amrex::Real c_R[5], dcRdT[5], e_RT[5];
  amrex::Real* eh_RT;
  if (HP == 1) {
    cp_R(c_R, T);
    dcvpRdT(dcRdT, T);
    eh_RT = &h_RT[0];
  } else {
    cv_R(c_R, T);
    dcvpRdT(dcRdT, T);
    speciesInternalEnergy(e_RT, T);
    eh_RT = &e_RT[0];
  }

  amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT = 0.0, dehmixdT = 0.0;
  for (int k = 0; k < 5; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[30 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 5; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 5; ++m) {
      dehmixdc += eh_RT[m] * J[k * 6 + m];
    }
    J[k * 6 + 5] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[35] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute an approx to the reaction Jacobian (for preconditioning)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT_SIMPLIFIED(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real* Tp, const int* HP)
{
  amrex::Real c[5];

  for (int k = 0; k < 5; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian_precond(J, c, *Tp, *HP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 5; k++) {
    J[30 + k] *= 1.e-6;
    J[k * 6 + 5] *= 1.e6;
  }
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int consP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 36; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[5];
  for (auto& val : wdot) {
    val = 0.0;
  }

  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 / T;
  amrex::Real refCinv = 1.0 / refC;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int k = 0; k < 5; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[5];
  gibbs(g_RT, T);

  // compute the species enthalpy
  amrex::Real h_RT[5];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[5];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 0: N2 + M <=> 2 N + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (4.2857 - 1) * sc[3] + (4.2857 - 1) * sc[4];
  // forward
  phi_f = sc[1];
  k_f = 7e+15 * exp(-1.6 * logT - (113199.092189437) * invT);
  dlnkfdT = -1.6 * invT + (113199.092189437) * invT2;
  // reverse
  phi_r = (sc[3] * sc[3]);
  Kc = refC * exp(g_RT[1] - 2.000000 * g_RT[3]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1]) + (2.000000 * h_RT[3]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q;     // N2
  wdot[3] += 2 * q; // N
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[N2]
    dqdci = +k_f;
    J[7] -= dqdci;     // dwdot[N2]/d[N2]
    J[9] += 2 * dqdci; // dwdot[N]/d[N2]
    // d()/d[N]
    dqdci = (4.2857 - 1) * q_nocor - k_r * 2.000000 * sc[3];
    J[19] -= dqdci;     // dwdot[N2]/d[N]
    J[21] += 2 * dqdci; // dwdot[N]/d[N]
    // d()/d[O]
    dqdci = (4.2857 - 1) * q_nocor;
    J[25] -= dqdci;     // dwdot[N2]/d[O]
    J[27] += 2 * dqdci; // dwdot[N]/d[O]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor + k_f;
    dqdc[2] = q_nocor;
    dqdc[3] = 4.2857 * q_nocor - k_r * 2.000000 * sc[3];
    dqdc[4] = 4.2857 * q_nocor;
    for (int k = 0; k < 5; k++) {
      J[6 * k + 1] -= dqdc[k];
      J[6 * k + 3] += 2 * dqdc[k];
    }
  }
  J[31] -= dqdT;     // dwdot[N2]/dT
  J[33] += 2 * dqdT; // dwdot[N]/dT

  // reaction 1: O2 + M <=> 2 O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (5 - 1) * sc[3] + (5 - 1) * sc[4];
  // forward
  phi_f = sc[0];
  k_f = 2e+15 * exp(-1.5 * logT - (59499.3321314184) * invT);
  dlnkfdT = -1.5 * invT + (59499.3321314184) * invT2;
  // reverse
  phi_r = (sc[4] * sc[4]);
  Kc = refC * exp(g_RT[0] - 2.000000 * g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0]) + (2.000000 * h_RT[4]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] -= q;     // O2
  wdot[4] += 2 * q; // O
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[O2]
    dqdci = +k_f;
    J[0] -= dqdci;     // dwdot[O2]/d[O2]
    J[4] += 2 * dqdci; // dwdot[O]/d[O2]
    // d()/d[N]
    dqdci = (5 - 1) * q_nocor;
    J[18] -= dqdci;     // dwdot[O2]/d[N]
    J[22] += 2 * dqdci; // dwdot[O]/d[N]
    // d()/d[O]
    dqdci = (5 - 1) * q_nocor - k_r * 2.000000 * sc[4];
    J[24] -= dqdci;     // dwdot[O2]/d[O]
    J[28] += 2 * dqdci; // dwdot[O]/d[O]
  } else {
    dqdc[0] = q_nocor + k_f;
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor;
    dqdc[3] = 5 * q_nocor;
    dqdc[4] = 5 * q_nocor - k_r * 2.000000 * sc[4];
    for (int k = 0; k < 5; k++) {
      J[6 * k + 0] -= dqdc[k];
      J[6 * k + 4] += 2 * dqdc[k];
    }
  }
  J[30] -= dqdT;     // dwdot[O2]/dT
  J[34] += 2 * dqdT; // dwdot[O]/dT

  // reaction 2: NO + M <=> N + O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (22 - 1) * sc[3] + (22 - 1) * sc[4];
  // forward
  phi_f = sc[2];
  k_f = 5000000000 * exp(-(75499.6092373453) * invT);
  dlnkfdT = (75499.6092373453) * invT2;
  // reverse
  phi_r = sc[3] * sc[4];
  Kc = refC * exp(g_RT[2] - g_RT[3] - g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2]) + (h_RT[3] + h_RT[4]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // NO
  wdot[3] += q; // N
  wdot[4] += q; // O
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[NO]
    dqdci = +k_f;
    J[14] -= dqdci; // dwdot[NO]/d[NO]
    J[15] += dqdci; // dwdot[N]/d[NO]
    J[16] += dqdci; // dwdot[O]/d[NO]
    // d()/d[N]
    dqdci = (22 - 1) * q_nocor - k_r * sc[4];
    J[20] -= dqdci; // dwdot[NO]/d[N]
    J[21] += dqdci; // dwdot[N]/d[N]
    J[22] += dqdci; // dwdot[O]/d[N]
    // d()/d[O]
    dqdci = (22 - 1) * q_nocor - k_r * sc[3];
    J[26] -= dqdci; // dwdot[NO]/d[O]
    J[27] += dqdci; // dwdot[N]/d[O]
    J[28] += dqdci; // dwdot[O]/d[O]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor + k_f;
    dqdc[3] = 22 * q_nocor - k_r * sc[4];
    dqdc[4] = 22 * q_nocor - k_r * sc[3];
    for (int k = 0; k < 5; k++) {
      J[6 * k + 2] -= dqdc[k];
      J[6 * k + 3] += dqdc[k];
      J[6 * k + 4] += dqdc[k];
    }
  }
  J[32] -= dqdT; // dwdot[NO]/dT
  J[33] += dqdT; // dwdot[N]/dT
  J[34] += dqdT; // dwdot[O]/dT

  // reaction 3: N2 + O <=> N + NO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[4];
  k_f = 640000000000 * exp(-1 * logT - (38399.4573342266) * invT);
  dlnkfdT = -1 * invT + (38399.4573342266) * invT2;
  // reverse
  phi_r = sc[2] * sc[3];
  Kc = exp(g_RT[1] - g_RT[2] - g_RT[3] + g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[4]) + (h_RT[2] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // N2
  wdot[2] += q; // NO
  wdot[3] += q; // N
  wdot[4] -= q; // O
  // d()/d[N2]
  dqdci = +k_f * sc[4];
  J[7] -= dqdci;  // dwdot[N2]/d[N2]
  J[8] += dqdci;  // dwdot[NO]/d[N2]
  J[9] += dqdci;  // dwdot[N]/d[N2]
  J[10] -= dqdci; // dwdot[O]/d[N2]
  // d()/d[NO]
  dqdci = -k_r * sc[3];
  J[13] -= dqdci; // dwdot[N2]/d[NO]
  J[14] += dqdci; // dwdot[NO]/d[NO]
  J[15] += dqdci; // dwdot[N]/d[NO]
  J[16] -= dqdci; // dwdot[O]/d[NO]
  // d()/d[N]
  dqdci = -k_r * sc[2];
  J[19] -= dqdci; // dwdot[N2]/d[N]
  J[20] += dqdci; // dwdot[NO]/d[N]
  J[21] += dqdci; // dwdot[N]/d[N]
  J[22] -= dqdci; // dwdot[O]/d[N]
  // d()/d[O]
  dqdci = +k_f * sc[1];
  J[25] -= dqdci; // dwdot[N2]/d[O]
  J[26] += dqdci; // dwdot[NO]/d[O]
  J[27] += dqdci; // dwdot[N]/d[O]
  J[28] -= dqdci; // dwdot[O]/d[O]
  // d()/dT
  J[31] -= dqdT; // dwdot[N2]/dT
  J[32] += dqdT; // dwdot[NO]/dT
  J[33] += dqdT; // dwdot[N]/dT
  J[34] -= dqdT; // dwdot[O]/dT

  // reaction 4: NO + O <=> N + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[4];
  k_f = 8400000 * exp(-(19449.8273500182) * invT);
  dlnkfdT = (19449.8273500182) * invT2;
  // reverse
  phi_r = sc[0] * sc[3];
  Kc = exp(-g_RT[0] + g_RT[2] - g_RT[3] + g_RT[4]);
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[4]) + (h_RT[0] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[0] += q; // O2
  wdot[2] -= q; // NO
  wdot[3] += q; // N
  wdot[4] -= q; // O
  // d()/d[O2]
  dqdci = -k_r * sc[3];
  J[0] += dqdci; // dwdot[O2]/d[O2]
  J[2] -= dqdci; // dwdot[NO]/d[O2]
  J[3] += dqdci; // dwdot[N]/d[O2]
  J[4] -= dqdci; // dwdot[O]/d[O2]
  // d()/d[NO]
  dqdci = +k_f * sc[4];
  J[12] += dqdci; // dwdot[O2]/d[NO]
  J[14] -= dqdci; // dwdot[NO]/d[NO]
  J[15] += dqdci; // dwdot[N]/d[NO]
  J[16] -= dqdci; // dwdot[O]/d[NO]
  // d()/d[N]
  dqdci = -k_r * sc[0];
  J[18] += dqdci; // dwdot[O2]/d[N]
  J[20] -= dqdci; // dwdot[NO]/d[N]
  J[21] += dqdci; // dwdot[N]/d[N]
  J[22] -= dqdci; // dwdot[O]/d[N]
  // d()/d[O]
  dqdci = +k_f * sc[2];
  J[24] += dqdci; // dwdot[O2]/d[O]
  J[26] -= dqdci; // dwdot[NO]/d[O]
  J[27] += dqdci; // dwdot[N]/d[O]
  J[28] -= dqdci; // dwdot[O]/d[O]
  // d()/dT
  J[30] += dqdT; // dwdot[O2]/dT
  J[32] -= dqdT; // dwdot[NO]/dT
  J[33] += dqdT; // dwdot[N]/dT
  J[34] -= dqdT; // dwdot[O]/dT

  amrex::Real c_R[5], dcRdT[5], e_RT[5];
  amrex::Real* eh_RT;
  if (consP == 1) {
    cp_R(c_R, T);
    dcvpRdT(dcRdT, T);
    eh_RT = &h_RT[0];
  } else {
    cv_R(c_R, T);
    dcvpRdT(dcRdT, T);
    speciesInternalEnergy(e_RT, T);
    eh_RT = &e_RT[0];
  }

  amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT = 0.0, dehmixdT = 0.0;
  for (int k = 0; k < 5; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[30 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 5; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 5; ++m) {
      dehmixdc += eh_RT[m] * J[k * 6 + m];
    }
    J[k * 6 + 5] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[35] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT(
  amrex::Real* J,
  const amrex::Real* sc,
  const amrex::Real* Tp,
  const int* consP)
{
  amrex::Real c[5];

  for (int k = 0; k < 5; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian(J, c, *Tp, *consP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 5; k++) {
    J[30 + k] *= 1.e-6;
    J[k * 6 + 5] *= 1.e6;
  }
}

// Transport function declarations

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENIMC(int* LENIMC)
{
  *LENIMC = 20;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENRMC(int* LENRMC)
{
  *LENRMC = 610;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNO(int* NO)
{
  *NO = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKK(int* KK)
{
  *KK = 5;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLITE(int* NLITE)
{
  *NLITE = 0;
}

// Patm in ergs/cm3
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPATM(amrex::Real* PATM)
{
  *PATM = 0.1013250000000000E+07;
}

// the molecular weights in g/mol
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetWT(amrex::Real* WT)
{
  WT[0] = 3.19988000E+01;
  WT[1] = 2.80134000E+01;
  WT[2] = 3.00061000E+01;
  WT[3] = 1.40067000E+01;
  WT[4] = 1.59994000E+01;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetEPS(amrex::Real* EPS)
{
  EPS[0] = 1.07400000E+02;
  EPS[1] = 9.75300000E+01;
  EPS[2] = 9.75300000E+01;
  EPS[3] = 7.14000000E+01;
  EPS[4] = 8.00000000E+01;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetSIG(amrex::Real* SIG)
{
  SIG[0] = 3.45800000E+00;
  SIG[1] = 3.62100000E+00;
  SIG[2] = 3.62100000E+00;
  SIG[3] = 3.29800000E+00;
  SIG[4] = 2.75000000E+00;
}

// the dipole moment in Debye
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetDIP(amrex::Real* DIP)
{
  DIP[0] = 0.00000000E+00;
  DIP[1] = 0.00000000E+00;
  DIP[2] = 0.00000000E+00;
  DIP[3] = 0.00000000E+00;
  DIP[4] = 0.00000000E+00;
}

// the polarizability in cubic Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPOL(amrex::Real* POL)
{
  POL[0] = 1.60000000E+00;
  POL[1] = 1.76000000E+00;
  POL[2] = 1.76000000E+00;
  POL[3] = 0.00000000E+00;
  POL[4] = 0.00000000E+00;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetZROT(amrex::Real* ZROT)
{
  ZROT[0] = 3.80000000E+00;
  ZROT[1] = 4.00000000E+00;
  ZROT[2] = 4.00000000E+00;
  ZROT[3] = 0.00000000E+00;
  ZROT[4] = 0.00000000E+00;
}

// 0: monoatomic, 1: linear, 2: nonlinear
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLIN(int* NLIN)
{
  NLIN[0] = 1;
  NLIN[1] = 1;
  NLIN[2] = 1;
  NLIN[3] = 0;
  NLIN[4] = 0;
}

// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFETA(amrex::Real* COFETA)
{
  COFETA[0] = -1.66882093E+01;
  COFETA[1] = 2.43208669E+00;
  COFETA[2] = -2.31031334E-01;
  COFETA[3] = 9.92288903E-03;
  COFETA[4] = -1.61980213E+01;
  COFETA[5] = 2.18577454E+00;
  COFETA[6] = -1.99190099E-01;
  COFETA[7] = 8.55173507E-03;
  COFETA[8] = -1.61636624E+01;
  COFETA[9] = 2.18577454E+00;
  COFETA[10] = -1.99190099E-01;
  COFETA[11] = 8.55173507E-03;
  COFETA[12] = -1.49200166E+01;
  COFETA[13] = 1.65982700E+00;
  COFETA[14] = -1.33228638E-01;
  COFETA[15] = 5.80456600E-03;
  COFETA[16] = -1.48292920E+01;
  COFETA[17] = 1.77514807E+00;
  COFETA[18] = -1.46750632E-01;
  COFETA[19] = 6.32347691E-03;
}

// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFLAM(amrex::Real* COFLAM)
{
  COFLAM[0] = -1.04646616E+00;
  COFLAM[1] = 2.57972663E+00;
  COFLAM[2] = -2.36689475E-01;
  COFLAM[3] = 1.03634023E-02;
  COFLAM[4] = 6.11011321E+00;
  COFLAM[5] = -5.38302077E-01;
  COFLAM[6] = 2.11666265E-01;
  COFLAM[7] = -1.10963682E-02;
  COFLAM[8] = 1.77153510E+00;
  COFLAM[9] = 1.29338101E+00;
  COFLAM[10] = -4.38439059E-02;
  COFLAM[11] = 6.52989271E-04;
  COFLAM[12] = 1.99829646E+00;
  COFLAM[13] = 1.65982700E+00;
  COFLAM[14] = -1.33228638E-01;
  COFLAM[15] = 5.80456600E-03;
  COFLAM[16] = 1.95600564E+00;
  COFLAM[17] = 1.77514807E+00;
  COFLAM[18] = -1.46750632E-01;
  COFLAM[19] = 6.32347691E-03;
}

// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFD(amrex::Real* COFD)
{
  COFD[0] = -1.51750939E+01;
  COFD[1] = 3.28612365E+00;
  COFD[2] = -2.08601363E-01;
  COFD[3] = 8.86791307E-03;
  COFD[4] = -1.48890251E+01;
  COFD[5] = 3.17402131E+00;
  COFD[6] = -1.94181586E-01;
  COFD[7] = 8.24913005E-03;
  COFD[8] = -1.46523799E+01;
  COFD[9] = 3.08143963E+00;
  COFD[10] = -1.82377255E-01;
  COFD[11] = 7.74645838E-03;
  COFD[12] = -1.49070512E+01;
  COFD[13] = 3.17402131E+00;
  COFD[14] = -1.94181586E-01;
  COFD[15] = 8.24913005E-03;
  COFD[16] = -1.46692643E+01;
  COFD[17] = 3.08143963E+00;
  COFD[18] = -1.82377255E-01;
  COFD[19] = 7.74645838E-03;
  COFD[20] = -1.46867388E+01;
  COFD[21] = 3.08143963E+00;
  COFD[22] = -1.82377255E-01;
  COFD[23] = 7.74645838E-03;
  COFD[24] = -1.38541766E+01;
  COFD[25] = 2.90828671E+00;
  COFD[26] = -1.60674476E-01;
  COFD[27] = 6.83840384E-03;
  COFD[28] = -1.36573795E+01;
  COFD[29] = 2.82664829E+00;
  COFD[30] = -1.50396283E-01;
  COFD[31] = 6.40764097E-03;
  COFD[32] = -1.36685721E+01;
  COFD[33] = 2.82664829E+00;
  COFD[34] = -1.50396283E-01;
  COFD[35] = 6.40764097E-03;
  COFD[36] = -1.30371931E+01;
  COFD[37] = 2.70807453E+00;
  COFD[38] = -1.37739569E-01;
  COFD[39] = 5.98112990E-03;
  COFD[40] = -1.39812430E+01;
  COFD[41] = 2.99911091E+00;
  COFD[42] = -1.72035408E-01;
  COFD[43] = 7.31245587E-03;
  COFD[44] = -1.37959275E+01;
  COFD[45] = 2.92157208E+00;
  COFD[46] = -1.62325668E-01;
  COFD[47] = 6.90681066E-03;
  COFD[48] = -1.38081461E+01;
  COFD[49] = 2.92157208E+00;
  COFD[50] = -1.62325668E-01;
  COFD[51] = 6.90681066E-03;
  COFD[52] = -1.30387966E+01;
  COFD[53] = 2.75419603E+00;
  COFD[54] = -1.42892228E-01;
  COFD[55] = 6.16664401E-03;
  COFD[56] = -1.29252037E+01;
  COFD[57] = 2.75556205E+00;
  COFD[58] = -1.41528678E-01;
  COFD[59] = 6.04017282E-03;
}

// List of specs with small weight, dim NLITE
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKTDIF(int* /*KTDIF*/)
{
}

// Poly fits for thermal diff ratios, dim NO*NLITE*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFTD(amrex::Real* /*COFTD*/)
{
}

// compute the critical parameters for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS(
  amrex::Real* Tci, amrex::Real* ai, amrex::Real* bi, amrex::Real* acentric_i)
{

  amrex::Real EPS[5];
  amrex::Real SIG[5];
  amrex::Real wt[5];
  amrex::Real Rcst = 83.144598; // in bar [CGS] !
  amrex::Real avogadro = 6.02214199e23;
  amrex::Real boltzmann = 1.3806503e-16; // we work in CGS

  egtransetEPS(EPS);
  egtransetSIG(SIG);
  get_mw(wt);

  // species 0: O2
  // Imported from NIST
  Tci[0] = 154.581000;
  ai[0] = 1e6 * 0.42748 * Rcst * Rcst * Tci[0] * Tci[0] /
          (31.998800 * 31.998800 * 50.430466);
  bi[0] = 0.08664 * Rcst * Tci[0] / (31.998800 * 50.430466);
  acentric_i[0] = 0.022200;

  // species 1: N2
  // Imported from NIST
  Tci[1] = 126.192000;
  ai[1] = 1e6 * 0.42748 * Rcst * Rcst * Tci[1] * Tci[1] /
          (28.013400 * 28.013400 * 33.958000);
  bi[1] = 0.08664 * Rcst * Tci[1] / (28.013400 * 33.958000);
  acentric_i[1] = 0.037200;

  // species 2: NO
  // Imported from NIST
  Tci[2] = 180.000000;
  ai[2] = 1e6 * 0.42748 * Rcst * Rcst * Tci[2] * Tci[2] /
          (30.006000 * 30.006000 * 64.800000);
  bi[2] = 0.08664 * Rcst * Tci[2] / (30.006000 * 64.800000);
  acentric_i[2] = 0.582000;

  // species 3: N
  Tci[3] = 1.316 * EPS[3];
  ai[3] = (5.55 * avogadro * avogadro * EPS[3] * boltzmann * 1e-24 * SIG[3] *
           SIG[3] * SIG[3]) /
          (wt[3] * wt[3]);
  bi[3] = 0.855 * avogadro * 1e-24 * SIG[3] * SIG[3] * SIG[3] / (wt[3]);
  acentric_i[3] = 0.0;

  // species 4: O
  Tci[4] = 1.316 * EPS[4];
  ai[4] = (5.55 * avogadro * avogadro * EPS[4] * boltzmann * 1e-24 * SIG[4] *
           SIG[4] * SIG[4]) /
          (wt[4] * wt[4]);
  bi[4] = 0.855 * avogadro * 1e-24 * SIG[4] * SIG[4] * SIG[4] / (wt[4]);
  acentric_i[4] = 0.0;
}

// compute the critical parameter quantities for each species for SRK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS_SRK(
  amrex::Real* sqrtOneOverTc,
  amrex::Real* sqrtAsti,
  amrex::Real* Bi,
  amrex::Real* Fomega)
{

  // species 0: O2
  // Imported from NIST
  sqrtOneOverTc[0] = 8.0430717653241e-02;
  sqrtAsti[0] = 3.6980081924740e+04;
  Bi[0] = 6.9005216444999e-01;
  Fomega[0] = 5.1945301904908e-01;

  // species 1: N2
  // Imported from NIST
  sqrtOneOverTc[1] = 8.9019282240563e-02;
  sqrtAsti[1] = 4.2022983929562e+04;
  Bi[1] = 9.5560052221563e-01;
  Fomega[1] = 5.4259343186608e-01;

  // species 2: NO
  // Imported from NIST
  sqrtOneOverTc[2] = 7.4535599249993e-02;
  sqrtAsti[2] = 4.0510614916903e+04;
  Bi[2] = 6.6687106751983e-01;
  Fomega[2] = 1.3368144381880e+00;

  // species 3: N
  // Computed from Lennard-Jones
  sqrtOneOverTc[3] = 1.0316275924140e-01;
  sqrtAsti[3] = 6.0232180334953e+04;
  Bi[3] = 1.3186612153657e+00;
  Fomega[3] = 4.8508000000000e-01;

  // species 4: O
  // Computed from Lennard-Jones
  sqrtOneOverTc[4] = 9.7460146948181e-02;
  sqrtAsti[4] = 4.2499093468641e+04;
  Bi[4] = 6.6928561533264e-01;
  Fomega[4] = 4.8508000000000e-01;
}

// gauss-jordan solver external routine
// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve_simplified(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort(
    "sgjsolve_simplified not implemented, choose a different solver ");
}

#endif
