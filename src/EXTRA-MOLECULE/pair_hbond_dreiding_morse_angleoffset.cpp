// clang-format off
/* ----------------------------------------------------------------------
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
   Contributing author: Tod A Pascal (Caltech), Don Xu/EiPi Fun
------------------------------------------------------------------------- */

#include "pair_hbond_dreiding_morse_angleoffset.h"

#include "atom.h"
#include "atom_vec.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "molecule.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

static constexpr int CHUNK = 8;

/* ---------------------------------------------------------------------- */

PairHbondDreidingMorseAngleoffset::PairHbondDreidingMorseAngleoffset(LAMMPS *lmp) :
  PairHbondDreidingMorse(lmp) {

  angle_offset_flag = 1;
}

/* ----------------------------------------------------------------------
 *    set coeffs for one or more type pairs
 * ---------------------------------------------------------------------- */

void PairHbondDreidingMorseAngleoffset::coeff(int narg, char **arg)
{
  if (narg < 7 || narg > 12)
    error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi,klo,khi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  utils::bounds_typelabel(FLERR, arg[2], 1, atom->ntypes, klo, khi, lmp, Atom::ATOM);

  int donor_flag;
  if (strcmp(arg[3],"i") == 0) donor_flag = 0;
  else if (strcmp(arg[3],"j") == 0) donor_flag = 1;
  else error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));

  double d0_one = utils::numeric(FLERR, arg[4], false, lmp);
  double alpha_one = utils::numeric(FLERR, arg[5], false, lmp);
  double r0_one = utils::numeric(FLERR, arg[6], false, lmp);

  int ap_one = ap_global;
  if (narg > 7) ap_one = utils::inumeric(FLERR, arg[7], false, lmp);
  double cut_inner_one = cut_inner_global;
  double cut_outer_one = cut_outer_global;
  if (narg > 9) {
    cut_inner_one = utils::numeric(FLERR, arg[8], false, lmp);
    cut_outer_one = utils::numeric(FLERR, arg[9], false, lmp);
  }
  if (cut_inner_one>cut_outer_one)
    error->all(FLERR,"Pair inner cutoff >= Pair outer cutoff");
  double cut_angle_one = cut_angle_global;
  if (narg > 10) cut_angle_one = utils::numeric(FLERR, arg[10], false, lmp) * MY_PI/180.0;
  double angle_offset_one = angle_offset_global;
  if (narg == 12) angle_offset_one = (180.0 - utils::numeric(FLERR, arg[11], false, lmp)) * MY_PI/180.0;
  if (angle_offset_one < 0.0 || angle_offset_one > 90.0 * MY_PI/180.0)
    error->all(FLERR,"Illegal angle offset {}", angle_offset_one);

  // grow params array if necessary

  if (nparams == maxparam) {
    maxparam += CHUNK;
    params = (Param *) memory->srealloc(params, maxparam*sizeof(Param),"pair:params");

    // make certain all addional allocated storage is initialized
    // to avoid false positives when checking with valgrind

    memset(params + nparams, 0, CHUNK*sizeof(Param));
  }

  params[nparams].d0 = d0_one;
  params[nparams].alpha = alpha_one;
  params[nparams].r0 = r0_one;
  params[nparams].ap = ap_one;
  params[nparams].cut_inner = cut_inner_one;
  params[nparams].cut_outer = cut_outer_one;
  params[nparams].cut_innersq = cut_inner_one*cut_inner_one;
  params[nparams].cut_outersq = cut_outer_one*cut_outer_one;
  params[nparams].cut_angle = cut_angle_one;
  params[nparams].angle_offset = angle_offset_one;
  params[nparams].denom_vdw = (params[nparams].cut_outersq-params[nparams].cut_innersq) *
    (params[nparams].cut_outersq-params[nparams].cut_innersq) *
    (params[nparams].cut_outersq-params[nparams].cut_innersq);

  // flag type2param with either i,j = D,A or j,i = D,A

  int count = 0;
  for (int i = ilo; i <= ihi; i++)
    for (int j = MAX(jlo,i); j <= jhi; j++)
      for (int k = klo; k <= khi; k++) {
        if (donor_flag == 0) type2param[i][j][k] = nparams;
        else type2param[j][i][k] = nparams;
        count++;
      }
  nparams++;

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));
}
