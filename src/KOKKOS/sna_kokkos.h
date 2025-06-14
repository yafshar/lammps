// clang-format off
/* -*- c++ -*- -------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Christian Trott (SNL), Stan Moore (SNL)
------------------------------------------------------------------------- */

#ifndef LMP_SNA_KOKKOS_H
#define LMP_SNA_KOKKOS_H

#include <complex>
#include <ctime>
#include <Kokkos_Core.hpp>
#include "kokkos_type.h"

#ifdef __SYCL_DEVICE_ONLY__
#include <CL/sycl.hpp>
#endif

namespace LAMMPS_NS {
// copied from pair_snap_kokkos.h
// pre-declare so sna_kokkos.h can refer to it
template<class DeviceType, typename real_type_, int vector_length_> class PairSNAPKokkos;
template<typename real_type_, int vector_length_>
struct WignerWrapper {
  using real_type = real_type_;
  using complex = SNAComplex<real_type>;
  static constexpr int vector_length = vector_length_;

  const int offset; // my offset into the vector (0, ..., vector_length - 1)
  real_type* buffer; // buffer of real numbers

  KOKKOS_INLINE_FUNCTION
  WignerWrapper(complex* buffer_, const int offset_)
   : offset(offset_), buffer(reinterpret_cast<real_type*>(buffer_))
  { ; }

  KOKKOS_INLINE_FUNCTION
  complex get(const int& ma) const {
    return complex(buffer[offset + 2 * vector_length * ma], buffer[offset + vector_length + 2 * vector_length * ma]);
  }

  KOKKOS_INLINE_FUNCTION
  void set(const int& ma, const complex& store) const {
    buffer[offset + 2 * vector_length * ma] = store.re;
    buffer[offset + vector_length + 2 * vector_length * ma] = store.im;
  }
};

struct alignas(8) FullHalfMapper {
  int idxu_half;
  int flip_sign; // 0 -> isn't flipped, 1 -> conj, -1 -> -conj
};

// Packed types for Zi, Yi lookup tables
// This is abstracted into a stand-alone struct so different implementations
// could be used for different architectures via various `ifdef` guards.
struct alignas(16) idxz_struct {
  reax_int4 j1_j2_j_jjuhalf;
  reax_int4 mabminmax;
  reax_int4 nanb_idxcg;

  idxz_struct() = default;

  KOKKOS_INLINE_FUNCTION
  idxz_struct(int j1, int j2, int j, int ma1min, int ma2max, int mb1min, int mb2max, int na, int nb, int jju_half, int idxcg)
    : j1_j2_j_jjuhalf{j1, j2, j, jju_half},
      mabminmax{ma1min, ma2max, mb1min, mb2max},
      nanb_idxcg{na, nb, idxcg, 0}
  { }

  KOKKOS_INLINE_FUNCTION
  void get_zi(int &j1, int &j2, int &j, int &ma1min, int &ma2max, int &mb1min, int &mb2max, int &na, int &nb, int &idxcg) {
    reax_int4 pack1 = this->j1_j2_j_jjuhalf;
    j1 = pack1.i0;
    j2 = pack1.i1;
    j = pack1.i2;
    reax_int4 pack2 = this->mabminmax;
    ma1min = pack2.i0;
    ma2max = pack2.i1;
    mb1min = pack2.i2;
    mb2max = pack2.i3;
    reax_int4 pack3 = this->nanb_idxcg;
    na = pack3.i0;
    nb = pack3.i1;
    idxcg = pack3.i2;
  }

  KOKKOS_INLINE_FUNCTION
  void get_yi(int &j1, int &j2, int &j, int &ma1min, int &ma2max, int &mb1min, int &mb2max, int &na, int &nb, int& jju_half, int& idxcg) {
    reax_int4 pack1 = this->j1_j2_j_jjuhalf;
    j1 = pack1.i0;
    j2 = pack1.i1;
    j = pack1.i2;
    jju_half = pack1.i3;
    reax_int4 pack2 = this->mabminmax;
    ma1min = pack2.i0;
    ma2max = pack2.i1;
    mb1min = pack2.i2;
    mb2max = pack2.i3;
    reax_int4 pack3 = this->nanb_idxcg;
    na = pack3.i0;
    nb = pack3.i1;
    idxcg = pack3.i2;
  }

  KOKKOS_INLINE_FUNCTION
  void get_yi_with_zlist(int &j1, int &j2, int &j, int &jju_half) {
    reax_int4 pack1 = this->j1_j2_j_jjuhalf;
    j1 = pack1.i0;
    j2 = pack1.i1;
    j = pack1.i2;
    jju_half = pack1.i3;
  }

};


template<class DeviceType, typename real_type_, int vector_length_>
class SNAKokkos {

 public:
  using real_type = real_type_;
  using complex = SNAComplex<real_type>;
  static constexpr int vector_length = vector_length_;

  using KKDeviceType = typename KKDevice<DeviceType>::value;
  static constexpr LAMMPS_NS::ExecutionSpace execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  static constexpr int host_flag = (execution_space == LAMMPS_NS::Host);

  typedef Kokkos::View<int*, DeviceType> t_sna_1i;
  typedef Kokkos::View<real_type*, DeviceType> t_sna_1d;
  typedef Kokkos::View<real_type*, KKDeviceType, Kokkos::MemoryTraits<Kokkos::Atomic>> t_sna_1d_atomic;
  typedef Kokkos::View<int**, DeviceType> t_sna_2i;
  typedef Kokkos::View<real_type**, DeviceType> t_sna_2d;
  typedef Kokkos::View<real_type**, Kokkos::LayoutLeft, DeviceType> t_sna_2d_ll;
  typedef Kokkos::View<real_type**, Kokkos::LayoutRight, DeviceType> t_sna_2d_lr;
  typedef Kokkos::View<real_type***, DeviceType> t_sna_3d;
  typedef Kokkos::View<real_type***, Kokkos::LayoutLeft, DeviceType> t_sna_3d_ll;
  typedef Kokkos::View<real_type***[3], DeviceType> t_sna_4d;
  typedef Kokkos::View<real_type****, Kokkos::LayoutLeft, DeviceType> t_sna_4d_ll;
  typedef Kokkos::View<real_type**[3], DeviceType> t_sna_3d3;
  typedef Kokkos::View<real_type*****, DeviceType> t_sna_5d;

  typedef Kokkos::View<complex*, DeviceType> t_sna_1c;
  typedef Kokkos::View<complex*, KKDeviceType, Kokkos::MemoryTraits<Kokkos::Atomic>> t_sna_1c_atomic;
  typedef Kokkos::View<complex**, DeviceType> t_sna_2c;
  typedef Kokkos::View<complex**, Kokkos::LayoutLeft, DeviceType> t_sna_2c_ll;
  typedef Kokkos::View<complex**, Kokkos::LayoutRight, DeviceType> t_sna_2c_lr;
  typedef Kokkos::View<complex***, DeviceType> t_sna_3c;
  typedef Kokkos::View<complex***, Kokkos::LayoutLeft, DeviceType> t_sna_3c_ll;
  typedef Kokkos::View<complex***[3], DeviceType> t_sna_4c;
  typedef Kokkos::View<complex***[3], DeviceType> t_sna_4c3;
  typedef Kokkos::View<complex****, Kokkos::LayoutLeft, DeviceType> t_sna_4c_ll;
  typedef Kokkos::View<complex**[3], DeviceType> t_sna_3c3;
  typedef Kokkos::View<complex*****, DeviceType> t_sna_5c;

  inline
  SNAKokkos() {};

  KOKKOS_INLINE_FUNCTION
  SNAKokkos(const SNAKokkos<DeviceType,real_type,vector_length>& sna, const typename Kokkos::TeamPolicy<DeviceType>::member_type& team);

  template<class CopyClass>
  inline
  SNAKokkos(const CopyClass&);

  KOKKOS_INLINE_FUNCTION
  ~SNAKokkos();

  inline
  void build_indexlist(); // SNAKokkos()

  inline
  void init();            //

  double memory_usage();

  int ncoeff;

  // functions for bispectrum coefficients, GPU only
  KOKKOS_INLINE_FUNCTION
  void compute_cayley_klein(const int&, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void pre_ui(const int&, const int&, const int&) const; // ForceSNAP

  // version of the code with parallelism over j_bend
  KOKKOS_INLINE_FUNCTION
  void compute_ui_small(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, const int, const int, const int, const int) const; // ForceSNAP
  // version of the code without parallelism over j_bend
  KOKKOS_INLINE_FUNCTION
  void compute_ui_large(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, const int, const int, const int) const; // ForceSNAP

  // desymmetrize ulisttot
  KOKKOS_INLINE_FUNCTION
  void transform_ui(const int&, const int&) const;

  template <bool chemsnap> KOKKOS_INLINE_FUNCTION
  void compute_zi(const int&, const int&) const;    // ForceSNAP
  template <bool chemsnap, bool need_atomics> KOKKOS_INLINE_FUNCTION
  void compute_yi(const int&, const int&) const; // ForceSNAP
  template <bool chemsnap, bool need_atomics> KOKKOS_INLINE_FUNCTION
  void compute_yi_with_zlist(const int&, const int&) const; // ForceSNAP
  template <bool chemsnap> KOKKOS_INLINE_FUNCTION
  void compute_bi(const int&, const int&) const;    // ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void compute_beta_linear(const int&, const int&, const int&) const;
  template <bool need_atomics> KOKKOS_INLINE_FUNCTION
  void compute_beta_quadratic(const int&, const int&, const int&) const;

  // functions for derivatives, GPU only
  // version of the code with parallelism over j_bend
  template<int dir>
  KOKKOS_INLINE_FUNCTION
  void compute_fused_deidrj_small(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, const int, const int, const int, const int) const; //ForceSNAP
  // version of the code without parallelism over j_bend
  template<int dir>
  KOKKOS_INLINE_FUNCTION
  void compute_fused_deidrj_large(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team, const int, const int, const int) const; //ForceSNAP

  // core "evaluation" functions that get plugged into "compute" functions
  // plugged into compute_ui_small, compute_ui_large
  KOKKOS_FORCEINLINE_FUNCTION
  void evaluate_ui_jbend(const WignerWrapper<real_type, vector_length>&, const complex&, const complex&, const real_type&, const int&,
                        const int&, const int&) const;
  // plugged into compute_zi, compute_yi
  KOKKOS_FORCEINLINE_FUNCTION
  complex evaluate_zi(const int&, const int&, const int&, const int&, const int&, const int&, const int&, const int&, const int&,
                        const int&, const int&, const int&, const real_type*) const;
  // plugged into compute_bi
  KOKKOS_FORCEINLINE_FUNCTION
  real_type evaluate_bi(const int&, const int&, const int&, const int&,
                          const int&, const int&, const int&) const;
  // plugged into compute_yi, compute_yi_with_zlist
  template <bool chemsnap> KOKKOS_FORCEINLINE_FUNCTION
  real_type evaluate_beta_scaled(const int&, const int&, const int&, const int&, const int&, const int&, const int&) const;
  // plugged into compute_fused_deidrj_small, compute_fused_deidrj_large
  KOKKOS_FORCEINLINE_FUNCTION
  real_type evaluate_duidrj_jbend(const WignerWrapper<real_type, vector_length>&, const complex&, const complex&, const real_type&,
                        const WignerWrapper<real_type, vector_length>&, const complex&, const complex&, const real_type&,
                        const int&, const int&, const int&) const;

  // functions for bispectrum coefficients, CPU only
  template <bool need_atomics> KOKKOS_INLINE_FUNCTION
  void compute_ui_cpu(const int&, const int&) const; // ForceSNAP

  // functions for derivatives, CPU only
  KOKKOS_INLINE_FUNCTION
  void compute_duidrj_cpu(const int&, const int&) const; //ForceSNAP
  KOKKOS_INLINE_FUNCTION
  void compute_deidrj_cpu(const int&, const int&) const; // ForceSNAP

  KOKKOS_INLINE_FUNCTION
  real_type compute_sfac(real_type, real_type, real_type, real_type) const; // add_uarraytot, compute_duarray

  KOKKOS_INLINE_FUNCTION
  real_type compute_dsfac(real_type, real_type, real_type, real_type) const; // compute_duarray

  KOKKOS_INLINE_FUNCTION
  void compute_s_dsfac(const real_type, const real_type, const real_type, const real_type, real_type&, real_type&) const; // compute_cayley_klein

#ifdef TIMING_INFO
  double* timers;
  timespec starttime, endtime;
  int print;
  int counter;
#endif

  //per sna class instance for OMP use

  // Per InFlight Particle
  t_sna_3d rij;
  t_sna_2i inside;
  t_sna_2d wj;
  t_sna_2d rcutij;
  t_sna_2d sinnerij;
  t_sna_2d dinnerij;
  t_sna_2i element;
  t_sna_3d dedr;
  int natom, natom_pad, nmax;

  void grow_rij(int, int);

  int twojmax, diagonalstyle;

  // Input beta coefficients; aliases the object in PairSnapKokkos
  t_sna_2d_lr d_coeffelem;

  // Beta for all atoms in list; aliases the object in PairSnapKokkos
  // for qSNAP the quadratic terms get accumulated into it
  // in compute_bi
  t_sna_2d d_beta;

  // Structures for both the CPU, GPU backend
  t_sna_3d ulisttot_re;
  t_sna_3d ulisttot_im;
  t_sna_3c ulisttot; // un-folded ulisttot

  t_sna_3c zlist;
  t_sna_3d blist;

  t_sna_3d ylist_re;
  t_sna_3d ylist_im;

  // Structures for the CPU backend only
  t_sna_3c ulist_cpu;
  t_sna_4c3 dulist_cpu;

  // Modified structures for GPU backend
  t_sna_2c a_gpu; // Cayley-Klein `a`
  t_sna_2c b_gpu; // `b`
  t_sna_3c da_gpu; // `da`
  t_sna_3c db_gpu; // `db`
  t_sna_3d sfac_gpu; // sfac, dsfac_{x,y,z}

  int idxcg_max, idxu_max, idxu_half_max, idxu_cache_max, idxz_max, idxb_max;

  // Chem snap counts
  int nelements;
  int ndoubles;
  int ntriples;

 private:
  real_type rmin0, rfac0;

  //use indexlist instead of loops, constructor generates these
  // Same across all SNAKokkos
  Kokkos::View<idxz_struct*, DeviceType> idxz;
  Kokkos::View<int*[3], DeviceType> idxb;
  Kokkos::View<int***, DeviceType> idxcg_block;

 public:
  Kokkos::View<int*, DeviceType> idxu_block;
  Kokkos::View<int*, DeviceType> idxu_half_block;
  Kokkos::View<int*, DeviceType> idxu_cache_block;
  Kokkos::View<FullHalfMapper*, DeviceType> idxu_full_half;

 private:
  Kokkos::View<int***, DeviceType> idxz_block;
  Kokkos::View<int***, DeviceType> idxb_block;

  // data for bispectrum coefficients

  // Same across all SNAKokkos
  t_sna_1d cglist;
  t_sna_2d rootpqarray;

  static const int nmaxfactorial = 167;
  static const double nfac_table[];
  inline
  double factorial(int);

  KOKKOS_INLINE_FUNCTION
  void create_team_scratch_arrays(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team); // SNAKokkos()
  KOKKOS_INLINE_FUNCTION
  void create_thread_scratch_arrays(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team); // SNAKokkos()

  inline
  void init_clebsch_gordan(); // init()

  inline
  void init_rootpqarray();    // init()

  inline
  double deltacg(int, int, int);  // init_clebsch_gordan

  inline
  int compute_ncoeff();           // SNAKokkos()

  // Sets the style for the switching function
  // 0 = none
  // 1 = cosine
  int switch_flag;

  // Sets the style for the inner switching function
  // 0 = none
  // 1 = cosine
  int switch_inner_flag;

  // Chem snap flags
  int chem_flag;
  int bnorm_flag;

  // Self-weight
  real_type wself;
  int wselfall_flag;

  // quadratic flag
  int quadratic_flag;

  int bzero_flag; // 1 if bzero subtracted from barray
  Kokkos::View<real_type*, DeviceType> bzero; // array of B values for isolated atoms
};

}

#include "sna_kokkos_impl.h"
#endif
