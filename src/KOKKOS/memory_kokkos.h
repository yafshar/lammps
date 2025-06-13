/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
// clang-format off

#ifndef LMP_MEMORY_KOKKOS_H
#define LMP_MEMORY_KOKKOS_H

#include "memory.h"             // IWYU pragma: export
#include "kokkos_type.h"

namespace LAMMPS_NS {

typedef MemoryKokkos MemKK;

class MemoryKokkos : public Memory {
 public:
  MemoryKokkos(class LAMMPS *lmp) : Memory(lmp) {}

/* ----------------------------------------------------------------------
   Kokkos versions of create/grow/destroy multi-dimensional arrays
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   create a 1d array
------------------------------------------------------------------------- */

template <typename TYPE>
TYPE create_kokkos(TYPE &data, typename TYPE::value_type *&array, int n1, const char *name)
{
  data = TYPE(name,n1);
  array = data.h_view.data();
  return data;
}

template <typename TYPE, typename HTYPE>
  TYPE create_kokkos(TYPE &data, HTYPE &h_data, typename TYPE::value_type *&array, int n1,
                     const char *name)
{
  data = TYPE(std::string(name),n1);
  h_data = Kokkos::create_mirror_view(data);
  array = h_data.data();
  return data;
}


template <typename TYPE, typename HTYPE>
  TYPE create_kokkos(TYPE &data, HTYPE &h_data, int n1, const char *name)
{
  data = TYPE(std::string(name),n1);
  h_data = Kokkos::create_mirror_view(data);
  return data;
}

/* ----------------------------------------------------------------------
   grow or shrink a 1d array
------------------------------------------------------------------------- */

template <typename TYPE>
TYPE grow_kokkos(TYPE &data, typename TYPE::value_type *&array, int n1, const char *name)
{
  if (array == nullptr) return create_kokkos(data,array,n1,name);

  data.resize(n1);
  array = data.h_view.data();
  return data;
}

/* ----------------------------------------------------------------------
   destroy a 1d array
------------------------------------------------------------------------- */

template <typename TYPE>
void destroy_kokkos(TYPE data, typename TYPE::value_type* &array)
{
  if (array == nullptr) return;
  data = TYPE();
  array = nullptr;
}

/* ----------------------------------------------------------------------
   create a 2d array
------------------------------------------------------------------------- */

template <typename TYPE, typename HTYPE>
  TYPE create_kokkos(TYPE &data, HTYPE &h_data, int n1, int n2, const char *name)
{
  data = TYPE(std::string(name),n1,n2);
  h_data = Kokkos::create_mirror_view(data);

  return data;
}

template <typename TYPE>
TYPE create_kokkos(TYPE &data, typename TYPE::value_type **&array,
                   int n1, int n2, const char *name)
{
  data = TYPE(std::string(name),n1,n2);
  bigint nbytes = ((bigint) sizeof(typename TYPE::value_type *)) * n1;
  array = (typename TYPE::value_type **) smalloc(nbytes,name);

  for (int i = 0; i < n1; i++) {
    if (n2 == 0)
      array[i] = nullptr;
    else
      array[i] = &data.h_view(i,0);
  }
  return data;
}

/* ----------------------------------------------------------------------
   create a 4d array with indices 2,3,4 offset, but not first
   2nd index from n2lo to n2hi inclusive
   3rd index from n3lo to n3hi inclusive
   4th index from n4lo to n4hi inclusive
   cannot grow it
------------------------------------------------------------------------- */

template <typename TYPE>
TYPE create4d_offset_kokkos(TYPE &data, typename TYPE::value_type ****&array,
                             int n1, int n2lo, int n2hi, int n3lo, int n3hi, int n4lo, int n4hi,
                             const char *name)
{
  //if (n1 <= 0 || n2lo > n2hi || n3lo > n3hi || n4lo > n4hi) array =  nullptr;

  printf("^^^^^ memoryKK->create_4d_offset_kokkos\n");

  int n2 = n2hi - n2lo + 1;
  int n3 = n3hi - n3lo + 1;
  int n4 = n4hi - n4lo + 1;
  data = TYPE(std::string(name),n1,n2,n3,n4);
  bigint nbytes = ((bigint) sizeof(typename TYPE::value_type ***)) * n1;
  array = (typename TYPE::value_type ****) smalloc(nbytes,name);

  for (int i = 0; i < n1; i++) {
    if (n2 == 0) {
      array[i] = nullptr;
    } else {
      nbytes = ((bigint) sizeof(typename TYPE::value_type **)) * n2;
      array[i] = (typename TYPE::value_type ***) smalloc(nbytes,name);
      for (int j = 0; j < n2; j++){
        if (n3 == 0){
          array[i][j] = nullptr;
        } else {
          nbytes = ((bigint) sizeof(typename TYPE::value_type *)) * n3;
          array[i][j] = (typename TYPE::value_type **) smalloc(nbytes, name);
          for (int k = 0; k < n3; k++){
            if (n4 == 0)
              array[i][j][k] = nullptr;
            else
              array[i][j][k] = &data.h_view(i,j,k,0);
          }
        }
      }
    }
  }

  return data;
}

template <typename TYPE, typename HTYPE>
  TYPE create_kokkos(TYPE &data, HTYPE &h_data,
                     typename TYPE::value_type **&array, int n1, int n2,
                     const char *name)
{
  data = TYPE(std::string(name),n1,n2);
  h_data = Kokkos::create_mirror_view(data);
  bigint nbytes = ((bigint) sizeof(typename TYPE::value_type *)) * n1;
  array = (typename TYPE::value_type **) smalloc(nbytes,name);

  for (int i = 0; i < n1; i++) {
    if (n2 == 0)
      array[i] = nullptr;
    else
      array[i] = &h_data(i,0);
  }
  return data;
}

template <typename TYPE>
TYPE create_kokkos(TYPE &data, typename TYPE::value_type **&array,
                   int n1, const char *name)
{
  data = TYPE(std::string(name),n1);
  bigint nbytes = ((bigint) sizeof(typename TYPE::value_type *)) * n1;
  array = (typename TYPE::value_type **) smalloc(nbytes,name);

  for (int i = 0; i < n1; i++)
    if (data.h_view.extent(1) == 0)
      array[i] = nullptr;
    else
      array[i] = &data.h_view(i,0);

  return data;
}

/* ----------------------------------------------------------------------
   grow or shrink a 2d array
------------------------------------------------------------------------- */

template <typename TYPE>
TYPE grow_kokkos(TYPE &data, typename TYPE::value_type **&array,
                 int n1, int n2, const char *name)
{
  if (array == nullptr) return create_kokkos(data,array,n1,n2,name);
  data.resize(n1,n2);
  bigint nbytes = ((bigint) sizeof(typename TYPE::value_type *)) * n1;
  array = (typename TYPE::value_type**) srealloc(array,nbytes,name);

  for (int i = 0; i < n1; i++)
    if (n2 == 0)
      array[i] = nullptr;
    else
      array[i] = &data.h_view(i,0);

  return data;
}

template <typename TYPE>
TYPE grow_kokkos(TYPE &data, typename TYPE::value_type **&array,
                 int n1, const char *name)
{
  if (array == nullptr) return create_kokkos(data,array,n1,name);

  data.resize(n1);

  bigint nbytes = ((bigint) sizeof(typename TYPE::value_type *)) * n1;
  array = (typename TYPE::value_type **) srealloc(array,nbytes,name);

  for (int i = 0; i < n1; i++)
    if (data.h_view.extent(1) == 0)
      array[i] = nullptr;
    else
      array[i] = &data.h_view(i,0);

  return data;
}

/* ----------------------------------------------------------------------
   destroy a 2d array
------------------------------------------------------------------------- */

template <typename TYPE>
void destroy_kokkos(TYPE data, typename TYPE::value_type** &array)
{
  if (array == nullptr) return;
  data = TYPE();
  sfree(array);
  array = nullptr;
}

/* ----------------------------------------------------------------------
   create a 3d array
------------------------------------------------------------------------- */

template <typename TYPE>
TYPE create_kokkos(TYPE &data, typename TYPE::value_type ***&array,
                   int n1, int n2, int n3, const char *name)
{
  data = TYPE(std::string(name),n1,n2,n3);
  bigint nbytes = ((bigint) sizeof(typename TYPE::value_type *)) * n1 * n2;
  typename TYPE::value_type **plane = (typename TYPE::value_type **) smalloc(nbytes,name);
  nbytes = ((bigint) sizeof(typename TYPE::value_type **)) * n1;
  array = (typename TYPE::value_type ***) smalloc(nbytes,name);

  bigint m;
  for (int i = 0; i < n1; i++) {
    if (n2 == 0) {
      array[i] = nullptr;
    } else {
      m = ((bigint) i) * n2;
      array[i] = &plane[m];

      for (int j = 0; j < n2; j++) {
        if (n3 == 0)
           array[i][j] = nullptr;
         else
           array[i][j] = &data.h_view(i,j,0);
      }
    }
  }
  return data;
}

template <typename TYPE, typename HTYPE>
  TYPE create_kokkos(TYPE &data, HTYPE &h_data,
                     typename TYPE::value_type ***&array, int n1, int n2, int n3,
                     const char *name)
{
  data = TYPE(std::string(name),n1,n2);
  h_data = Kokkos::create_mirror_view(data);
  bigint nbytes = ((bigint) sizeof(typename TYPE::value_type *)) * n1 * n2;
  typename TYPE::value_type **plane = (typename TYPE::value_type **) smalloc(nbytes,name);
  nbytes = ((bigint) sizeof(typename TYPE::value_type **)) * n1;
  array = (typename TYPE::value_type ***) smalloc(nbytes,name);

  bigint m;
  for (int i = 0; i < n1; i++) {
    if (n2 == 0) {
      array[i] = nullptr;
    } else {
      m = ((bigint) i) * n2;
      array[i] = &plane[m];

      for (int j = 0; j < n2; j++) {
        if (n3 == 0)
           array[i][j] = nullptr;
         else
           array[i][j] = &data.h_view(i,j,0);
      }
    }
  }
  return data;
}

template <typename TYPE, typename HTYPE>
  TYPE create_kokkos(TYPE &data, HTYPE &h_data, int n1, int n2, int n3,
                     const char *name)
{
  data = TYPE(std::string(name),n1,n2,n3);
  h_data = Kokkos::create_mirror_view(data);
  return data;
}


/* ----------------------------------------------------------------------
   grow or shrink a 3d array
------------------------------------------------------------------------- */

template <typename TYPE>
TYPE grow_kokkos(TYPE &data, typename TYPE::value_type ***&array,
                   int n1, int n2, int n3, const char *name)
{
  if (array == nullptr) return create_kokkos(data,array,n1,n2,n3,name);
  data.resize(n1,n2,n3);
  bigint nbytes = ((bigint) sizeof(typename TYPE::value_type *)) * n1 * n2;
  typename TYPE::value_type **plane = (typename TYPE::value_type **) srealloc(array[0],nbytes,name);
  nbytes = ((bigint) sizeof(typename TYPE::value_type **)) * n1;
  array = (typename TYPE::value_type ***) srealloc(array,nbytes,name);

  bigint m;
  for (int i = 0; i < n1; i++) {
    if (n2 == 0) {
      array[i] = nullptr;
    } else {
      m = ((bigint) i) * n2;
      array[i] = &plane[m];

      for (int j = 0; j < n2; j++) {
        if (n3 == 0)
           array[i][j] = nullptr;
         else
           array[i][j] = &data.h_view(i,j,0);
      }
    }
  }
  return data;
}

/* ----------------------------------------------------------------------
   destroy a 3d array
------------------------------------------------------------------------- */

template <typename TYPE>
void destroy_kokkos(TYPE data, typename TYPE::value_type*** &array)
{
  if (array == nullptr) return;
  data = TYPE();

  sfree(array[0]);
  sfree(array);
  array = nullptr;
}

/* ----------------------------------------------------------------------
   reallocate Kokkos views without initialization
   deallocate first to reduce memory use
   for the first case, enforce values are given for all dimensions
   for the second case, allow zero values given for dimensions
------------------------------------------------------------------------- */

template <typename TYPE, typename... Indices>
static std::enable_if_t<TYPE::rank_dynamic == sizeof...(Indices),void> realloc_kokkos(TYPE &data, const char *name, Indices... ns)
{
  data = TYPE();
  data = TYPE(std::string(name), ns...);
}

template <typename TYPE, typename... Indices>
static std::enable_if_t<TYPE::rank_dynamic == sizeof...(Indices) || sizeof...(Indices) == 0,void> realloc_kokkos_allow_zero(TYPE &data, const char *name, Indices... ns)
{
  data = TYPE();
  if constexpr (sizeof...(Indices) != 0)
    data = TYPE(std::string(name), ns...);
}

/* ----------------------------------------------------------------------
   get memory usage of Kokkos view in bytes
------------------------------------------------------------------------- */

template <typename TYPE>
static double memory_usage(TYPE &data)
{
  return data.span() * sizeof(typename TYPE::value_type);
}

/* ----------------------------------------------------------------------
  legacy functions
------------------------------------------------------------------------- */

template <typename TYPE>
TYPE destroy_kokkos(TYPE &data)
{
  data = TYPE();
  return data;
}

template <typename TYPE>
TYPE create_kokkos(TYPE &data, int n1, const char *name)
{
  data = TYPE();
  data = TYPE(std::string(name),n1);
  return data;
}

template <typename TYPE>
TYPE create_kokkos(TYPE &data, int n1, int n2, const char *name)
{
  data = TYPE();
  data = TYPE(std::string(name),n1,n2);
  return data;
}

template <typename TYPE>
TYPE create_kokkos(TYPE &data, int n1, int n2, int n3 ,const char *name)
{
  data = TYPE();
  data = TYPE(std::string(name),n1,n2,n3);
  return data;
}

template <typename TYPE>
TYPE create_kokkos(TYPE &data, int n1, int n2, int n3, int n4 ,const char *name)
{
  data = TYPE();
  data = TYPE(std::string(name),n1,n2,n3,n4);
  return data;
}

template <typename TYPE>
TYPE create_kokkos(TYPE &data, int n1, int n2, int n3, int n4, int n5 ,const char *name)
{
  data = TYPE();
  data = TYPE(std::string(name),n1,n2,n3,n4,n5);
  return data;
}

template <typename TYPE>
TYPE create_kokkos(TYPE &data, int n1, int n2, int n3, int n4, int n5 , int n6 ,const char *name)
{
  data = TYPE();
  data = TYPE(std::string(name),n1,n2,n3,n4,n5,n6);
  return data;
}

};

}

#endif
