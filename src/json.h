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

#ifndef LMP_JSON_H
#define LMP_JSON_H

// wrapper around including the JSON parsing and writing class
// Do NOT include in any header file

#include "nlohmann/json.hpp"

namespace LAMMPS_NS {
using json = ::nlohmann_lmp::json;
}
#endif
