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
   Contributing authors: Axel Kohlmeyer  (Temple U),
                         Ryan S. Elliott (UMN),
                         Ellad B. Tadmor (UMN),
                         Ronald Miller   (Carleton U),
                         Yaser Afshar    (UMN)
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

#include "kim_interactions.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_store_kim.h"
#include "force.h"
#include "improper.h"
#include "improper_hybrid.h"
#include "input.h"
#include "label_map.h"
#include "modify.h"
#include "tokenizer.h"
#include "update.h"
#include "variable.h"

#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" {
#include "KIM_SimulatorHeaders.h"
}

using namespace LAMMPS_NS;

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

void KimInteractions::command(int narg, char **arg)
{
  if (input->variable->retrieve("kim_bonded_ff")) {
    if (narg > 0) error->all(FLERR, "Illegal 'kim interactions' command");
  } else if (narg < 1) utils::missing_cmd_args(FLERR, "kim interactions", error);

  if (atom->lmap) {
    // If the atom type labels have been defined, kim interactions should not accept arguments

    auto lmap = atom->lmap;

    // Atom Type Label

    if (!lmap->is_complete(Atom::ATOM))
      error->all(FLERR, "Label map is incomplete. All types must be assigned a type label.");

    // Bond Type Label

    if (lmap->nbondtypes && !lmap->is_complete(Atom::BOND))
      error->all(FLERR, "Label map is incomplete. All bond types must be assigned a bond label.");

    // Angle Type Label

    if (lmap->nangletypes && !lmap->is_complete(Atom::ANGLE))
      error->all(FLERR, "Label map is incomplete. All angle types must be assigned an "
                 "angle label.");

    // Dihedral Type Label

    if (lmap->ndihedraltypes && !lmap->is_complete(Atom::DIHEDRAL))
      error->all(FLERR, "Label map is incomplete. All dihedral types must be assigned a "
                 "dihedral label.");

    // Improper Type Label

    if (lmap->nimpropertypes && !lmap->is_complete(Atom::IMPROPER))
      error->all(FLERR, "Label map is incomplete. All improper types must be assigned an "
                 "improper label.");
  }

  if (!domain->box_exist)
    error->all(FLERR, "Use of 'kim interactions' before simulation box is defined");

  do_setup(narg, arg);
}

/* ---------------------------------------------------------------------- */

void KimInteractions::do_setup(int narg, char **arg)
{
  bool fixed_types;
  if (narg == 0) {
    fixed_types = true;
  } else {
    const std::string arg_str(arg[0]);
    if ((narg == 1) && (arg_str == "fixed_types")) {
      fixed_types = true;
    } else if (narg != atom->ntypes) {
      error->all(FLERR, "Illegal 'kim interactions' command.\nThe LAMMPS simulation has {} atom "
                 "type(s), but {} chemical species passed to the 'kim interactions' command",
                 atom->ntypes, narg);
    } else {
      fixed_types = false;
    }
  }

  char *model_name = nullptr;
  KIM_SimulatorModel *simulatorModel(nullptr);

  // check if we had a kim init command by finding fix STORE/KIM
  // retrieve model name and pointer to simulator model class instance.
  // validate model name if not given as null pointer.

  auto fix_store = dynamic_cast<FixStoreKIM *>(modify->get_fix_by_id("KIM_MODEL_STORE"));
  if (fix_store) {
    model_name = (char *)fix_store->getptr("model_name");
    simulatorModel = (KIM_SimulatorModel *)fix_store->getptr("simulator_model");
  } else error->all(FLERR, "Must use 'kim init' before 'kim interactions'");

  // Begin output to log file
  input->write_echo("#=== BEGIN kim interactions ==================================\n");

  if (simulatorModel) {
    auto first_visit = input->variable->find("kim_update");
    if (!fixed_types) {
      std::string atom_type_sym_list =
        fmt::format("{}", fmt::join(arg, arg + narg, " "));

      std::string atom_type_num_list =
        fmt::format("{}", species_to_atomic_no(arg[0]));

      for (int i = 1; i < narg; ++i)
        atom_type_num_list += fmt::format(" {}", species_to_atomic_no(arg[i]));

      KIM_SimulatorModel_AddTemplateMap(
          simulatorModel, "atom-type-sym-list", atom_type_sym_list.c_str());
      KIM_SimulatorModel_AddTemplateMap(
          simulatorModel, "atom-type-num-list", atom_type_num_list.c_str());
      KIM_SimulatorModel_CloseTemplateMap(simulatorModel);

      // validate species selection

      int sim_num_species;
      bool species_is_supported;
      char const *sim_species;
      KIM_SimulatorModel_GetNumberOfSupportedSpecies(
          simulatorModel, &sim_num_species);

      for (auto atom_type_sym : utils::split_words(atom_type_sym_list)) {
        species_is_supported = false;
        if (atom_type_sym == "NULL") continue;
        for (int i = 0; i < sim_num_species; ++i) {
          KIM_SimulatorModel_GetSupportedSpecies(
            simulatorModel, i, &sim_species);
          if (atom_type_sym == sim_species) species_is_supported = true;
        }
        if (!species_is_supported) {
          error->all(FLERR, "Species '{}' is not supported by this KIM Simulator Model",
                     atom_type_sym);
        }
      }
    } else {
      KIM_SimulatorModel_CloseTemplateMap(simulatorModel);
    }

    // check if units are unchanged

    int sim_fields, sim_lines;
    const char *sim_field, *sim_value;
    KIM_SimulatorModel_GetNumberOfSimulatorFields(simulatorModel, &sim_fields);
    for (int i = 0; i < sim_fields; ++i) {
      KIM_SimulatorModel_GetSimulatorFieldMetadata(simulatorModel, i, &sim_lines, &sim_field);

      const std::string sim_field_str(sim_field);
      if (sim_field_str == "units") {
        KIM_SimulatorModel_GetSimulatorFieldLine(simulatorModel, i, 0, &sim_value);

        if (strcmp(sim_value, update->unit_style) != 0)
          error->all(FLERR, "Incompatible units for KIM Simulator Model: {} vs {}",
                     sim_value, update->unit_style);
      }
    }

    bool no_model_definition = true;
    for (int i = 0; i < sim_fields; ++i) {
      KIM_SimulatorModel_GetSimulatorFieldMetadata(
        simulatorModel, i, &sim_lines, &sim_field);

      const std::string sim_field_str(sim_field);
      if (sim_field_str == "model-defn") {
        if (first_visit < 0) input->one("variable kim_update equal 0");
        else input->one("variable kim_update equal 1");
        if (domain->periodicity[0] &&
            domain->periodicity[1] &&
            domain->periodicity[2])
          input->one("variable kim_periodic equal 1");
        else if (!domain->periodicity[0] &&
                 !domain->periodicity[1] &&
                 !domain->periodicity[2])
          input->one("variable kim_periodic equal 0");
        else input->one("variable kim_periodic equal 2");

        // KIM Simulator Model has a Model definition
        no_model_definition = false;

        for (int j = 0; j < sim_lines; ++j) {
          KIM_SimulatorModel_GetSimulatorFieldLine(simulatorModel, i, j, &sim_value);
          if (utils::strmatch(sim_value, "^KIM_SET_TYPE_PARAMETERS")) {
            // Notes regarding the KIM_SET_TYPE_PARAMETERS command
            //  * This is an INTERNAL command.
            //  * It is intended for use only by KIM Simulator Models.
            //  * It is not possible to use this command outside of the context
            //    of the kim interactions command and KIM Simulator Models.
            //  * The command performs a transformation from symbolic
            //    string-based atom types to lammps numeric atom types for
            //    the pair_coeff and charge settings.
            //  * The command is not documented fully as it is expected to be
            //    temporary.  Eventually it should be replaced by a more
            //    comprehensive symbolic types support in lammps.
            KIM_SET_TYPE_PARAMETERS(sim_value);
          } else {
            input->one(sim_value);
          }
        }
      }
    }

    if (no_model_definition)
      error->all(FLERR, "KIM Simulator Model has no Model definition");

    KIM_SimulatorModel_OpenAndInitializeTemplateMap(simulatorModel);

  } else {
    // not a simulator model. issue pair_style and pair_coeff commands.

    if (fixed_types)
      error->all(FLERR, "fixed_types cannot be used with a KIM Portable Model");

    // NOTE: all references to arg must appear before calls to input->one()
    // as that will reset the argument vector.

    auto cmd1 = fmt::format("pair_style kim {}", model_name);
    auto cmd2 = fmt::format("pair_coeff * * {}", fmt::join(arg, arg + narg, " "));

    input->one(cmd1);
    input->one(cmd2);
  }

  // End output to log file
  input->write_echo("#=== END kim interactions ====================================\n\n");
}

/* ---------------------------------------------------------------------- */

class KIMLabelMap : protected Pointers {
 public:
  KIMLabelMap(LAMMPS *lmp) : Pointers(lmp) {};
  ~KIMLabelMap() = default;

  std::vector<std::string> read_file(const std::string &filename) const
  {
    FILE *fp{nullptr};
    if (comm->me == 0) {
      fp = fopen(filename.c_str(), "r");
      if (!fp)
        error->one(FLERR, "Parameter file '{}' not found", filename);
    }
    std::vector<std::string> lines;
    char *line = new char[MAXLINE];
    bool eof{false};
    while (!eof) {
      eof = utils::read_lines_from_file(fp, 1, MAXLINE, line, comm->me, world);
      auto trimmed = utils::trim_comment(line);
      if (trimmed.find_first_not_of(" \t\n\r") == std::string::npos) continue;
      lines.push_back(trimmed);
    }
    if (comm->me == 0)
      if (fp) fclose(fp);
    delete[] line;
    return lines;
  }

  void process_file(const std::string &filename)
  {
    // read the input file

    const auto lines = read_file(filename);

    // Create a set of atom species from bonded FF parameter file

    for (auto line : lines) {
      auto words = Tokenizer(line).as_vector();
      if (words[0] == "pair_coeff") {
        if ((utils::is_type(words[1]) != 1) || (utils::is_type(words[2]) != 1)) {
          const std::string str = (utils::is_type(words[1]) != 1) ? words[1] : words[2];
          if (isdigit(str[0]))
            error->all(FLERR, "Type label {} in the SM parameter file cannot start with a number",
                       str);
          if ((str[0] == '*') || (str[0] == '#'))
            error->all(FLERR, "Type label {} in the SM parameter file cannot start with a {} {}",
                       str, str[0], "character");
          error->all(FLERR, "The {} string in the SM parameter file is not a valid type label {}",
                     str, "string");
        }

        typelabel.insert(words[1]);
        typelabel.insert(words[2]);

        auto key = fmt::format("{} {}", words[1], words[2]);
        auto value = fmt::format("{}", fmt::join(words.begin() + 3, words.end(), " "));

        auto search = pair_coeff_map.find(key);
        if (search != pair_coeff_map.end()) {
          bool new_value = true;
          for (auto val : search->second) {
            if (val == value) {
              new_value = false;
              break;
            };
          }
          if (new_value) search->second.push_back(value);
        } else pair_coeff_map[key].push_back(value);
      }
    }

    // Check all the bonded coefficients for correctness and validity
    // & add bonded coefficients to the map

    for (auto line : lines) {
      auto words = Tokenizer(line).as_vector();
      if (words[0] == "bond_coeff") {
        auto twords = Tokenizer(words[1], "-").as_vector();
        if (twords.size() != 2)
          error->all(FLERR, "Wrong bond type {} in the SM parameter file", words[1]);

        for (auto word : twords) {
          auto search_atom = typelabel.find(word);
          if (search_atom == typelabel.end())
            error->all(FLERR, "Must specify the pairwise force field coefficients for {} {}", word,
                       "before bond_coeff in the SM parameter file");
        }

        auto value = fmt::format("{}", fmt::join(words.begin() + 2, words.end(), " "));

        auto search = bond_coeff_map.find(words[1]);
        if (search != bond_coeff_map.end()) {
          bool new_value = true;
          for (auto val : search->second) {
            if (val == value) {
              new_value = false;
              break;
            };
          }
          if (new_value) search->second.push_back(value);
        } else bond_coeff_map[words[1]].push_back(value);
      } else if (words[0] == "angle_coeff") {
        auto twords = Tokenizer(words[1], "-").as_vector();
        if (twords.size() != 3)
          error->all(FLERR, "Wrong angle type {} in the SM parameter file", words[1]);

        for (auto word : twords) {
          auto search_atom = typelabel.find(word);
          if (search_atom == typelabel.end())
            error->all(FLERR, "Must specify the pairwise force field coefficients for {} {}", word,
                       "before angle_coeff in the SM parameter file");
        }

        auto value = fmt::format("{}", fmt::join(words.begin() + 2, words.end(), " "));

        auto search = angle_coeff_map.find(words[1]);
        if (search != angle_coeff_map.end()) {
          bool new_value = true;
          for (auto val : search->second) {
            if (val == value) {
              new_value = false;
              break;
            };
          }
          if (new_value) search->second.push_back(value);
        } else angle_coeff_map[words[1]].push_back(value);

      } else if (words[0] == "dihedral_coeff") {
        auto twords = Tokenizer(words[1], "-").as_vector();
        if (twords.size() != 4)
          error->all(FLERR, "Wrong dihedral type {} in the SM parameter file", words[1]);

        for (auto word : twords) {
          auto search_atom = typelabel.find(word);
          if (search_atom == typelabel.end())
            error->all(FLERR, "Must specify the pairwise force field coefficients for {} {}", word,
                       "before dihedral_coeff in the SM parameter file");
        }

        auto value = fmt::format("{}", fmt::join(words.begin() + 2, words.end(), " "));

        auto search = dihedral_coeff_map.find(words[1]);
        if (search != dihedral_coeff_map.end()) {
          bool new_value = true;
          for (auto val : search->second) {
            if (val == value) {
              new_value = false;
              break;
            };
          }
          if (new_value) search->second.push_back(value);
        } else dihedral_coeff_map[words[1]].push_back(value);

      } else if (words[0] == "improper_coeff") {
        auto twords = Tokenizer(words[1], "-").as_vector();
        if (twords.size() != 4)
          error->all(FLERR, "Wrong improper type {} in the SM parameter file", words[1]);

        for (auto word : twords) {
          auto search_atom = typelabel.find(word);
          if (search_atom == typelabel.end())
            error->all(FLERR, "Must specify the pairwise force field coefficients for {} {}", word,
                       "before improper_coeff in the SM parameter file");
        }

        auto value = fmt::format("{}", fmt::join(words.begin() + 2, words.end(), " "));

        auto search = improper_coeff_map.find(words[1]);
        if (search != improper_coeff_map.end()) {
          bool new_value = true;
          for (auto val : search->second) {
            if (val == value) {
              new_value = false;
              break;
            };
          }
          if (new_value) search->second.push_back(value);
        } else improper_coeff_map[words[1]].push_back(value);

      } else if (words[0] != "pair_coeff")
        error->all(FLERR, "Invalid KEY {} in the SM parameter file", words[0]);
    }

    check_SM_parameter_file();

    natomtypes = static_cast<int>(typelabel.size());
    nbondtypes = static_cast<int>(bond_coeff_map.size());
    nangletypes = static_cast<int>(angle_coeff_map.size());
    ndihedraltypes = static_cast<int>(dihedral_coeff_map.size());
    nimpropertypes = static_cast<int>(improper_coeff_map.size());
  }

  void check_SM_parameter_file(){
    std::string emsg = "Symmetrically equivalent {} {} and {} have different coefficients "
                       "in the SM parameter file";

    for (auto it : pair_coeff_map) {
      auto twords = Tokenizer(it.first).as_vector();
      // symmetry for pair coeffs (A B ~ B A)
      auto key = fmt::format("{} {}", twords[1], twords[0]);
      if (it.first == key) continue;
      auto search = pair_coeff_map.find(key);
      if (search != pair_coeff_map.end()) {
        if (it.second.size() != search->second.size())
          error->all(FLERR, emsg, "pair_coeff", it.first, key);
        for (auto v1 = it.second.begin(), v2 = search->second.begin(); v1 != it.second.end();
             ++v1, ++v2) {
          if (*v1 != *v2) error->all(FLERR, emsg, "pair_coeff", it.first, key);
        }
      }
    }

    for (auto it : bond_coeff_map) {
      auto twords = Tokenizer(it.first, "-").as_vector();
      // symmetry for bond coeffs (A-B ~ B-A)
      auto key = fmt::format("{}-{}", twords[1], twords[0]);
      if (it.first == key) continue;
      auto search = bond_coeff_map.find(key);
      if (search != bond_coeff_map.end()) {
        if (it.second.size() != search->second.size())
          error->all(FLERR, emsg, "bond_coeff", it.first, key);
        for (auto v1 = it.second.begin(), v2 = search->second.begin(); v1 != it.second.end();
             ++v1, ++v2) {
          if (*v1 != *v2) error->all(FLERR, emsg, "bond_coeff", it.first, key);
        }
      }
    }

    for (auto it : angle_coeff_map) {
      auto twords = Tokenizer(it.first, "-").as_vector();
      // symmetry for angle coeffs (A-B-C ~ C-B-A)
      auto key = fmt::format("{}-{}-{}", twords[2], twords[1], twords[0]);
      if (it.first == key) continue;
      auto search = angle_coeff_map.find(key);
      if (search != angle_coeff_map.end()) {
        if (it.second.size() != search->second.size())
          error->all(FLERR, emsg, "angle_coeff", it.first, key);
        for (auto v1 = it.second.begin(), v2 = search->second.begin(); v1 != it.second.end();
             ++v1, ++v2) {
          if (*v1 != *v2) error->all(FLERR, emsg, "angle_coeff", it.first, key);
        }
      }
    }

    for (auto it : dihedral_coeff_map) {
      auto twords = Tokenizer(it.first, "-").as_vector();
      // symmetry for dihedral coeffs (A-B-C-D ~ D-C-B-A)
      auto key = fmt::format("{}-{}-{}-{}", twords[3], twords[2], twords[1], twords[0]);
      if (it.first == key) continue;
      auto search = dihedral_coeff_map.find(key);
      if (search != dihedral_coeff_map.end()) {
        if (it.second.size() != search->second.size())
          error->all(FLERR, emsg, "dihedral_coeff", it.first, key);
        for (auto v1 = it.second.begin(), v2 = search->second.begin(); v1 != it.second.end();
             ++v1, ++v2) {
          if (*v1 != *v2) error->all(FLERR, emsg, "dihedral_coeff", it.first, key);
        }
      }
    }

    if (nimpropertypes) {
      ImproperHybrid *hybrid = nullptr;
      int nstyles = 1;

      if (utils::strmatch(force->improper_style, "^hybrid")) {
        hybrid = dynamic_cast<ImproperHybrid *>(force->improper);
        nstyles = hybrid->nstyles;
      }

      for (auto it : improper_coeff_map) {
        auto twords = Tokenizer(it.first, "-").as_vector();

        for (int style = 0; style < nstyles; ++style) {
          auto improper = hybrid ? hybrid->styles[style] : force->improper;
          // Find the non symmetry indices
          std::vector<int> nonsymind;
          for (int i = 0; i < 4; ++i)
            if (improper->symmatoms[i] == 0) nonsymind.push_back(i);
          if (nonsymind.size() == 2) {
            constexpr int d[2] = {1, 0};
            std::string key = "";
            for (int j = 0, c = 0; j < 4; ++j) {
              if (improper->symmatoms[j] == 0) key += twords[nonsymind[d[c++]]];
              else key += twords[j];
              if (j != 3) key += "-";
            }
            if (it.first == key) continue;
            auto search = improper_coeff_map.find(key);
            if (search != improper_coeff_map.end()) {
              if (it.second.size() != search->second.size())
                error->all(FLERR, emsg, "improper_coeff", it.first, key);
              for (auto v1 = it.second.begin(), v2 = search->second.begin(); v1 != it.second.end();
                   ++v1, ++v2) {
                if (*v1 != *v2) error->all(FLERR, emsg, "improper_coeff", it.first, key);
              }
            }
          } else {
            constexpr int d[5][3] = {{1, 0, 2}, {2, 0, 1}, {0, 2, 1}, {1, 2, 0}, {2, 1, 0}};
            for (int i = 0; i < 5; ++i) {
              std::string key = "";
              for (int j = 0, c = 0; j < 4; ++j) {
                if (improper->symmatoms[j] == 0) key += twords[nonsymind[d[i][c++]]];
                else key += twords[j];
                if (j != 3) key += "-";
              }
              if (it.first == key) continue;
              auto search = improper_coeff_map.find(key);
              if (search != improper_coeff_map.end()) {
                if (it.second.size() != search->second.size())
                  error->all(FLERR, emsg, "improper_coeff", it.first, key);
                for (auto v1 = it.second.begin(), v2 = search->second.begin(); v1 != it.second.end();
                     ++v1, ++v2) {
                  if (*v1 != *v2) error->all(FLERR, emsg, "improper_coeff", it.first, key);
                }
              }
            }
          }
        }
      }
    }
  }

  int natomtypes{0};      // number of atom types
  int nbondtypes{0};      // number of bond types with no symmetry
  int nangletypes{0};     // number of angle types with no symmetry
  int ndihedraltypes{0};  // number of dihedral types with no symmetry
  int nimpropertypes{0};  // number of improper dihedral types with no symmetry

  std::unordered_set<std::string> typelabel;

  // An array is used in the map for special cases like class2 (e.g. angle_coeff 1, angle_coeff 1 bb)
  std::unordered_map<std::string, std::vector<std::string>> pair_coeff_map;
  std::unordered_map<std::string, std::vector<std::string>> bond_coeff_map;
  std::unordered_map<std::string, std::vector<std::string>> angle_coeff_map;
  std::unordered_map<std::string, std::vector<std::string>> dihedral_coeff_map;
  std::unordered_map<std::string, std::vector<std::string>> improper_coeff_map;
};

/* ---------------------------------------------------------------------- */

void KimInteractions::KIM_SET_TYPE_PARAMETERS(const std::string &input_line) const
{
  auto words = utils::split_words(input_line);

  const std::string set_key = words[1];
  if (set_key != "pair" && set_key != "charge" && set_key != "bonded_ff")
    error->all(FLERR, "Unrecognized KEY {} for KIM_SET_TYPE_PARAMETERS command", set_key);

  if (set_key == "bonded_ff") {
    if (words[2] != "lammps")
      error->all(FLERR, "Unrecognized format {} for KIM_SET_TYPE_PARAMETERS command", words[2]);

    if (!atom->labelmapflag)
      error->all(FLERR, "Use of 'kim interactions' before setting the 'type labels'");

    // Create a KIMLabelMap object, read and process the parameter file

    KIMLabelMap klmap(lmp);

    klmap.process_file(words[3]);

    auto lmap = atom->lmap;

    std::string emsg = "{} Type Label {} is not defined in the SM parameter file";

    // Atom Type Label

    for (auto tlb : lmap->typelabel) {
      auto search = klmap.typelabel.find(tlb);
      if (search == klmap.typelabel.end())
        error->all(FLERR, emsg, "Atom", tlb);
    }

    // Consider symmetry for pair coeffs (A B ~ B A)

    for (auto tlb1 : lmap->typelabel) {
      for (auto tlb2 : lmap->typelabel) {
        auto key = fmt::format("{} {}", tlb1, tlb2);
        auto search = klmap.pair_coeff_map.find(key);
        if (search != klmap.pair_coeff_map.end()) {
          // pair_coeff command can override a previous setting for the same I,J pair
          for (auto val : search->second) {
            input->one(fmt::format("pair_coeff {} {}", key, val));
          }
        }
      }
    }

    // Bond Type Label

    for (auto btlb : lmap->btypelabel) {
      auto search = klmap.bond_coeff_map.find(btlb);
      if (search == klmap.bond_coeff_map.end()) {

        // Add bond_coeff based on symmetry for bond coeffs (A-B ~ B-A)

        auto twords = Tokenizer(btlb, "-").as_vector();
        auto key = fmt::format("{}-{}", twords[1], twords[0]);

        if (key == btlb) error->all(FLERR, emsg, "Bond", btlb);

        auto search = klmap.bond_coeff_map.find(key);
        if (search == klmap.bond_coeff_map.end()) {
          error->all(FLERR, emsg, "Bond", btlb);
        } else {
          // bond_coeff command can override a previous setting for the same bond type
          for (auto val : search->second)
            input->one(fmt::format("bond_coeff {} {}", btlb, val));
        }
      } else {
        // bond_coeff command can override a previous setting for the same bond type
        for (auto val : search->second)
          input->one(fmt::format("bond_coeff {} {}", btlb, val));
      }
    }

    // Angle Type Label

    for (auto atlb : lmap->atypelabel) {
      auto search = klmap.angle_coeff_map.find(atlb);
      if (search == klmap.angle_coeff_map.end()) {

        // Add angle_coeff based on symmetry for angle coeffs (A-B-C ~ C-B-A)

        auto twords = Tokenizer(atlb, "-").as_vector();
        auto key = fmt::format("{}-{}-{}", twords[2], twords[1], twords[0]);

        if (key == atlb) error->all(FLERR, emsg, "Angle", atlb);

        search = klmap.angle_coeff_map.find(key);
        if (search == klmap.angle_coeff_map.end()) {
          error->all(FLERR, emsg, "Angle", atlb);
        } else {
          // angle_coeff command can override a previous setting for the same angle type
          for (auto val : search->second)
            input->one(fmt::format("angle_coeff {} {}", atlb, val));
        }
      } else {
        // angle_coeff command can override a previous setting for the same angle type
        for (auto val : search->second)
          input->one(fmt::format("angle_coeff {} {}", atlb, val));
      }
    }

    // Dihedral Type Label

    for (auto dtlb : lmap->dtypelabel) {
      auto search = klmap.dihedral_coeff_map.find(dtlb);
      if (search == klmap.dihedral_coeff_map.end()) {

        // Add dihedral_coeff based on symmetry for dihedral coeffs (A-B-C-D ~ D-C-B-A)

        auto twords = Tokenizer(dtlb, "-").as_vector();
        auto key = fmt::format("{}-{}-{}-{}", twords[3], twords[2], twords[1], twords[0]);

        if (key == dtlb) error->all(FLERR, emsg, "Dihedral", dtlb);

        search = klmap.dihedral_coeff_map.find(key);
        if (search == klmap.dihedral_coeff_map.end()) {
          error->all(FLERR, emsg, "Dihedral", dtlb);
        } else {
          // dihedral_coeff command can override a previous setting for the same dihedral type
          for (auto val : search->second)
            input->one(fmt::format("dihedral_coeff {} {}", dtlb, val));
        }
      } else {
        // dihedral_coeff command can override a previous setting for the same dihedral type
        for (auto val : search->second)
          input->one(fmt::format("dihedral_coeff {} {}", dtlb, val));
      }
    }

    // Improper Type Label
    if (klmap.nimpropertypes) {
      ImproperHybrid *hybrid = nullptr;
      int nstyles = 1;

      if (utils::strmatch(force->improper_style, "^hybrid")) {
        hybrid = dynamic_cast<ImproperHybrid *>(force->improper);
        nstyles = hybrid->nstyles;
      }

      for (auto itlb : lmap->itypelabel)  {
        auto search = klmap.improper_coeff_map.find(itlb);
        if (search == klmap.improper_coeff_map.end()) {
          // Add improper_coeff based on symmetry for improper coeffs

          auto twords = Tokenizer(itlb, "-").as_vector();
          bool found = false;
          for (int style = 0; style < nstyles; ++style) {
            auto improper = hybrid ? hybrid->styles[style] : force->improper;

            // Find the non symmetry indices
            std::vector<int> nonsymind;
            for (int i = 0; i < 4; ++i)
              if (improper->symmatoms[i] == 0) nonsymind.push_back(i);

            found = false;

            if (nonsymind.size() == 2) {
              constexpr int d[2] = {1, 0};
              std::string key = "";
              for (int j = 0, c = 0; j < 4; ++j) {
                if (improper->symmatoms[j] == 0) key += twords[nonsymind[d[c++]]];
                else key += twords[j];
                if (j != 3) key += "-";
              }

              if (key == itlb) continue;

              search = klmap.improper_coeff_map.find(key);
              if (search != klmap.improper_coeff_map.end()) {
                // improper_coeff command can override a previous setting for the same improper type
                for (auto val : search->second)
                  input->one(fmt::format("improper_coeff {} {}", itlb, val));
                found = true;
              }
            } else {
              constexpr int d[5][3] = {{1, 0, 2}, {2, 0, 1}, {0, 2, 1}, {1, 2, 0}, {2, 1, 0}};
              for (int i = 0; i < 5; ++i) {
                std::string key = "";
                for (int j = 0, c = 0; j < 4; ++j) {
                  if (improper->symmatoms[j] == 0) key += twords[nonsymind[d[i][c++]]];
                  else key += twords[j];
                  if (j != 3) key += "-";
                }

                if (key == itlb) continue;

                search = klmap.improper_coeff_map.find(key);
                if (search != klmap.improper_coeff_map.end()) {
                  // improper_coeff command can override a previous setting for the same improper type
                  for (auto val : search->second)
                    input->one(fmt::format("improper_coeff {} {}", itlb, val));
                  found = true;
                  break;
                }
              }
            }

            if (found) break;
          }

          if (!found) error->all(FLERR, emsg, "Improper", itlb);
        } else {
          // improper_coeff command can override a previous setting for the same improper type
          for (auto val : search->second)
            input->one(fmt::format("improper_coeff {} {}", itlb, val));
        }
      }
    }

  } else {
    std::vector<std::string> species;
    if (input->variable->retrieve("kim_bonded_ff")) {
      if (words.size() > 3)
        error->all(FLERR, "Unrecognized KEY {} for KIM_SET_TYPE_PARAMETERS command in bonded SMs",
                   fmt::join(words.begin() + 3, words.end(), " "));
    } else {
      for (auto s = words.begin() + 3; s != words.end(); ++s) species.emplace_back(*s);
      if ((int)species.size() != atom->ntypes)
        error->all(FLERR, "KIM_SET_TYPE_PARAMETERS command species {} do not match with {}{}",
                   fmt::join(species.begin(), species.end(), " "), atom->ntypes, " system ntypes");

      for (int ia = 0; ia < atom->ntypes; ++ia)
        if (!(atom->lmap && (atom->lmap->find(species[ia],Atom::ATOM) == ia + 1)))
          input->one(fmt::format("labelmap atom {} {}", ia + 1, species[ia]));
    }

    std::string filename = words[2];
    FILE *fp = nullptr;
    if (comm->me == 0) {
      fp = fopen(filename.c_str(), "r");
      if (fp == nullptr) error->one(FLERR, "Parameter file {} not found", filename);
    }

    char line[MAXLINE], *ptr;
    int n, eof = 0;

    while (true) {
      if (comm->me == 0) {
        ptr = fgets(line, MAXLINE,fp);
        if (!ptr) {
          eof = 1;
          fclose(fp);
        } else n = strlen(line) + 1;
      }
      MPI_Bcast(&eof, 1, MPI_INT, 0, world);
      if (eof) break;
      MPI_Bcast(&n, 1, MPI_INT, 0, world);
      MPI_Bcast(line, n, MPI_CHAR, 0, world);

      auto trimmed = utils::trim_comment(line);
      if (trimmed.find_first_not_of(" \t\n\r") == std::string::npos) continue;

      words = utils::split_words(trimmed);
      if (set_key == "pair") {
        if (species.empty()) {
          input->one(fmt::format("pair_coeff {}", trimmed));
        } else {
          for (int ia = 0; ia < atom->ntypes; ++ia) {
            for (int ib = ia; ib < atom->ntypes; ++ib)
              if (((species[ia] == words[0]) && (species[ib] == words[1]))
                || ((species[ib] == words[0]) && (species[ia] == words[1])))
                input->one(fmt::format("pair_coeff {}", trimmed));
          }
        }
      } else {
        if (species.empty()) {
          input->one(fmt::format("set type {} charge {}", words[0], words[1]));
        } else {
          for (int ia = 0; ia < atom->ntypes; ++ia)
            if (species[ia] == words[0])
              input->one(fmt::format("set type {} charge {}", words[0], words[1]));
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int KimInteractions::species_to_atomic_no(const std::string &species) const
{
  if (species == "H") return 1;
  else if (species == "He") return 2;
  else if (species == "Li") return 3;
  else if (species == "Be") return 4;
  else if (species == "B") return 5;
  else if (species == "C") return 6;
  else if (species == "N") return 7;
  else if (species == "O") return 8;
  else if (species == "F") return 9;
  else if (species == "Ne") return 10;
  else if (species == "Na") return 11;
  else if (species == "Mg") return 12;
  else if (species == "Al") return 13;
  else if (species == "Si") return 14;
  else if (species == "P") return 15;
  else if (species == "S") return 16;
  else if (species == "Cl") return 17;
  else if (species == "Ar") return 18;
  else if (species == "K") return 19;
  else if (species == "Ca") return 20;
  else if (species == "Sc") return 21;
  else if (species == "Ti") return 22;
  else if (species == "V") return 23;
  else if (species == "Cr") return 24;
  else if (species == "Mn") return 25;
  else if (species == "Fe") return 26;
  else if (species == "Co") return 27;
  else if (species == "Ni") return 28;
  else if (species == "Cu") return 29;
  else if (species == "Zn") return 30;
  else if (species == "Ga") return 31;
  else if (species == "Ge") return 32;
  else if (species == "As") return 33;
  else if (species == "Se") return 34;
  else if (species == "Br") return 35;
  else if (species == "Kr") return 36;
  else if (species == "Rb") return 37;
  else if (species == "Sr") return 38;
  else if (species == "Y") return 39;
  else if (species == "Zr") return 40;
  else if (species == "Nb") return 41;
  else if (species == "Mo") return 42;
  else if (species == "Tc") return 43;
  else if (species == "Ru") return 44;
  else if (species == "Rh") return 45;
  else if (species == "Pd") return 46;
  else if (species == "Ag") return 47;
  else if (species == "Cd") return 48;
  else if (species == "In") return 49;
  else if (species == "Sn") return 50;
  else if (species == "Sb") return 51;
  else if (species == "Te") return 52;
  else if (species == "I") return 53;
  else if (species == "Xe") return 54;
  else if (species == "Cs") return 55;
  else if (species == "Ba") return 56;
  else if (species == "La") return 57;
  else if (species == "Ce") return 58;
  else if (species == "Pr") return 59;
  else if (species == "Nd") return 60;
  else if (species == "Pm") return 61;
  else if (species == "Sm") return 62;
  else if (species == "Eu") return 63;
  else if (species == "Gd") return 64;
  else if (species == "Tb") return 65;
  else if (species == "Dy") return 66;
  else if (species == "Ho") return 67;
  else if (species == "Er") return 68;
  else if (species == "Tm") return 69;
  else if (species == "Yb") return 70;
  else if (species == "Lu") return 71;
  else if (species == "Hf") return 72;
  else if (species == "Ta") return 73;
  else if (species == "W") return 74;
  else if (species == "Re") return 75;
  else if (species == "Os") return 76;
  else if (species == "Ir") return 77;
  else if (species == "Pt") return 78;
  else if (species == "Au") return 79;
  else if (species == "Hg") return 80;
  else if (species == "Tl") return 81;
  else if (species == "Pb") return 82;
  else if (species == "Bi") return 83;
  else if (species == "Po") return 84;
  else if (species == "At") return 85;
  else if (species == "Rn") return 86;
  else if (species == "Fr") return 87;
  else if (species == "Ra") return 88;
  else if (species == "Ac") return 89;
  else if (species == "Th") return 90;
  else if (species == "Pa") return 91;
  else if (species == "U") return 92;
  else if (species == "Np") return 93;
  else if (species == "Pu") return 94;
  else if (species == "Am") return 95;
  else if (species == "Cm") return 96;
  else if (species == "Bk") return 97;
  else if (species == "Cf") return 98;
  else if (species == "Es") return 99;
  else if (species == "Fm") return 100;
  else if (species == "Md") return 101;
  else if (species == "No") return 102;
  else if (species == "Lr") return 103;
  else if (species == "Rf") return 104;
  else if (species == "Db") return 105;
  else if (species == "Sg") return 106;
  else if (species == "Bh") return 107;
  else if (species == "Hs") return 108;
  else if (species == "Mt") return 109;
  else if (species == "Ds") return 110;
  else if (species == "Rg") return 111;
  else if (species == "Cn") return 112;
  else if (species == "Nh") return 113;
  else if (species == "Fl") return 114;
  else if (species == "Mc") return 115;
  else if (species == "Lv") return 116;
  else if (species == "Ts") return 117;
  else if (species == "Og") return 118;
  else return -1;
}
