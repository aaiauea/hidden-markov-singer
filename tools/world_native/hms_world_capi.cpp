// HMS minimal C ABI around the WORLD vocoder (https://github.com/mmorise/World).
//
// Why this exists: the reference WORLD implementation is C++, is not packaged on
// every platform, and its Python bindings need Python.h.  This wrapper exposes
// WORLD through a flat, caller-allocated C ABI so HMS can talk to the *real*
// WORLD vocoder over ctypes with no Python build dependencies.
//
// Build:  tools/build_world.sh   (produces libhms_world.so)
// Python: hms/vocoder/world_native.py
//
// All arrays are row-major doubles; the caller owns every buffer.

#include <cstddef>
#include <vector>

#include "world/cheaptrick.h"
#include "world/d4c.h"
#include "world/dio.h"
#include "world/harvest.h"
#include "world/stonemask.h"
#include "world/synthesis.h"

extern "C" {

// ---------------------------------------------------------------- geometry --

int hms_f0_length(int fs, int x_length, double frame_period) {
  return GetSamplesForDIO(fs, x_length, frame_period);
}

int hms_fft_size(int fs) {
  CheapTrickOption option;
  InitializeCheapTrickOption(fs, &option);
  return GetFFTSizeForCheapTrick(fs, &option);
}

int hms_synth_length(int f0_length, int fs, double frame_period) {
  return static_cast<int>(f0_length * frame_period / 1000.0 * fs);
}

// --------------------------------------------------------------------- F0 ----
// alg: 0 = DIO + StoneMask (default, fast), 1 = Harvest + StoneMask (robust, slow)

int hms_f0(const double* x, int x_length, int fs, double frame_period,
           double f0_floor, double f0_ceil, int alg, int refine,
           double* temporal_positions, double* f0) {
  const int f0_length = GetSamplesForDIO(fs, x_length, frame_period);
  if (alg == 1) {
    HarvestOption option;
    InitializeHarvestOption(&option);
    if (f0_floor > 0) option.f0_floor = f0_floor;
    if (f0_ceil > 0) option.f0_ceil = f0_ceil;
    option.frame_period = frame_period;
    Harvest(x, x_length, fs, &option, temporal_positions, f0);
  } else {
    DioOption option;
    InitializeDioOption(&option);
    if (f0_floor > 0) option.f0_floor = f0_floor;
    if (f0_ceil > 0) option.f0_ceil = f0_ceil;
    option.frame_period = frame_period;
    Dio(x, x_length, fs, &option, temporal_positions, f0);
  }
  if (refine) {
    std::vector<double> refined(f0_length);
    StoneMask(x, x_length, fs, temporal_positions, f0, f0_length, refined.data());
    for (int i = 0; i < f0_length; ++i) f0[i] = refined[i];
  }
  return f0_length;
}

// -------------------------------------------------- spectral envelope (sp) ---

int hms_spectral_envelope(const double* x, int x_length, int fs,
                          const double* temporal_positions, const double* f0,
                          int f0_length, double frame_period, double* sp) {
  CheapTrickOption option;
  InitializeCheapTrickOption(fs, &option);
  const int fft_size = GetFFTSizeForCheapTrick(fs, &option);
  option.f0_floor = GetF0FloorForCheapTrick(fs, fft_size);
  option.fft_size = fft_size;
  const int bins = fft_size / 2 + 1;

  std::vector<double*> rows(f0_length);
  for (int i = 0; i < f0_length; ++i) rows[i] = sp + static_cast<size_t>(i) * bins;
  CheapTrick(x, x_length, fs, temporal_positions, f0, f0_length, &option,
             rows.data());
  return bins;
}

// ------------------------------------------------------ aperiodicity (ap) ----

int hms_aperiodicity(const double* x, int x_length, int fs,
                     const double* temporal_positions, const double* f0,
                     int f0_length, double frame_period, double* ap) {
  CheapTrickOption ct_option;
  InitializeCheapTrickOption(fs, &ct_option);
  const int fft_size = GetFFTSizeForCheapTrick(fs, &ct_option);
  const int bins = fft_size / 2 + 1;

  D4COption option;
  InitializeD4COption(&option);
  std::vector<double*> rows(f0_length);
  for (int i = 0; i < f0_length; ++i) rows[i] = ap + static_cast<size_t>(i) * bins;
  D4C(x, x_length, fs, temporal_positions, f0, f0_length, fft_size, &option,
      rows.data());
  return bins;
}

// ------------------------------------------------------------- synthesis -----

void hms_synthesize(const double* f0, int f0_length, const double* sp,
                    const double* ap, int fft_size, double frame_period, int fs,
                    int y_length, double* y) {
  const int bins = fft_size / 2 + 1;
  std::vector<const double*> sp_rows(f0_length);
  std::vector<const double*> ap_rows(f0_length);
  for (int i = 0; i < f0_length; ++i) {
    sp_rows[i] = sp + static_cast<size_t>(i) * bins;
    ap_rows[i] = ap + static_cast<size_t>(i) * bins;
  }
  Synthesis(f0, f0_length, sp_rows.data(), ap_rows.data(), fft_size,
            frame_period, fs, y_length, y);
}

}  // extern "C"
