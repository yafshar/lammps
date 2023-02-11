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
   Contributing authors: Yaser Afshar (UMN)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the Free
   Software Foundation; either version 2 of the License, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
   more details.

   You should have received a copy of the GNU General Public License along with
   this program; if not, see <https://www.gnu.org/licenses>.

   Linking LAMMPS statically or dynamically with other modules is making a
   combined work based on LAMMPS. Thus, the terms and conditions of the GNU
   General Public License cover the whole combination.

   In addition, as a special exception, the copyright holders of LAMMPS give
   you permission to combine LAMMPS with free software programs or libraries
   that are released under the GNU LGPL and with code included in the standard
   release of the "kim-api" under the CDDL (or modified versions of such code,
   with unchanged license). You may copy and distribute such a system following
   the terms of the GNU GPL for LAMMPS and the licenses of the other code
   concerned, provided that you include the source code of that other code
   when and as the GNU GPL requires distribution of source code.

   Note that people who make modified versions of LAMMPS are not obligated to
   grant this special exception for their modified versions; it is their choice
   whether to do so. The GNU General Public License gives permission to release
   a modified version without this exception; this exception also makes it
   possible to release a modified version which carries forward this exception.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Designed for use with the kim-api-2.1.0 (and newer) package
------------------------------------------------------------------------- */

#include "kim_create_box.h"

#include "atom.h"
#include "error.h"
#include "fix_store_kim.h"
#include "input.h"
#include "modify.h"
#include "variable.h"

extern "C" {
#include "KIM_SimulatorHeaders.h"
}

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

KimCreateBox::KimCreateBox(LAMMPS *lmp) : Pointers(lmp) {}

/* ---------------------------------------------------------------------- */

void KimCreateBox::command(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR, "Illegal kim create_box command");

  // check if we had a kim init command by finding fix STORE/KIM

  int ifix = modify->find_fix("KIM_MODEL_STORE");
  if (ifix < 0) error->all(FLERR, "Must use 'kim init' before 'kim create_box'");

  auto tmp = input->variable->retrieve("kim_bonded_ff");
  if (!tmp) error->all(FLERR, "Can not use 'kim create_box' for non-bonded SMs");

  // Begin output to log file

  input->write_echo("#=== BEGIN kim create_box ====================================\n");

  // set atom and topology type quantities

  tmp = input->variable->retrieve("kim_natomtypes");
  int ntypes = utils::inumeric(FLERR, tmp, false, lmp);
  if (ntypes < 0) error->all(FLERR, "Illegal KIM simulator");

  auto cmd = fmt::format("create_box {} {}", ntypes, arg[0]);

  tmp = input->variable->retrieve("kim_nbondtypes");
  if (tmp) {
    int nbondtypes = utils::inumeric(FLERR, tmp, false, lmp);
    if (nbondtypes < 0) error->all(FLERR, "Illegal KIM simulator");
    cmd += fmt::format(" bond/types {}", nbondtypes);
  }

  tmp = input->variable->retrieve("kim_nangletypes");
  if (tmp) {
    int nangletypes = utils::inumeric(FLERR, tmp, false, lmp);
    if (nangletypes < 0) error->all(FLERR, "Illegal KIM simulator");
    cmd += fmt::format(" angle/types {}", nangletypes);
  }

  tmp = input->variable->retrieve("kim_ndihedraltypes");
  if (tmp) {
    int ndihedraltypes = utils::inumeric(FLERR, tmp, false, lmp);
    if (ndihedraltypes < 0) error->all(FLERR, "Illegal KIM simulator");
    cmd += fmt::format(" dihedral/types {}", ndihedraltypes);
  }

  tmp = input->variable->retrieve("kim_nimpropertypes");
  if (tmp) {
    int nimpropertypes = utils::inumeric(FLERR, tmp, false, lmp);
    if (nimpropertypes < 0) error->all(FLERR, "Illegal KIM simulator");
    cmd += fmt::format(" improper/types {}", nimpropertypes);
  }

  // check optional args

  int iarg = 1;
  while (iarg < narg) {
    const std::string sarg(arg[iarg]);
    if (sarg == "extra/bond/per/atom") {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal 'kim create_box' command");
      cmd += fmt::format(" extra/bond/per/atom {}", arg[iarg + 1]);
    } else if (sarg == "extra/angle/per/atom") {
      if (iarg + 2 > narg) error->all(FLERR,"Illegal 'kim create_box' command");
      cmd += fmt::format(" extra/angle/per/atom {}", arg[iarg + 1]);
    } else if (sarg == "extra/dihedral/per/atom") {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal 'kim create_box' command");
      cmd += fmt::format(" extra/dihedral/per/atom {}", arg[iarg + 1]);
    } else if (sarg == "extra/improper/per/atom") {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal 'kim create_box' command");
      cmd += fmt::format(" extra/improper/per/atom {}", arg[iarg + 1]);
    } else if (sarg == "extra/special/per/atom") {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal 'kim create_box' command");
      cmd += fmt::format(" extra/special/per/atom {}", arg[iarg + 1]);
    } else error->all(FLERR, "Illegal 'kim create_box' command");
    iarg += 2;
  }

  input->one(cmd);

  // set all masses to a default of 1.0

  double *masses = new double[ntypes + 1];
  for (int itype = 1; itype <= ntypes; ++itype) masses[itype] = 1.0;
  atom->set_mass(masses);
  delete [] masses;

  // End output to log file

  input->write_echo("#=== END kim create_box ======================================\n");
}
