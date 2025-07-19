// unit tests for neighbor list functionality

#include "library.h"

#include "info.h"
#include "utils.h"

#include "../testing/core.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

using ::testing::ContainsRegex;

// whether to print verbose output (i.e. not capturing LAMMPS screen output).
bool verbose = false;

namespace LAMMPS_NS {

class NeighborListsBin : public LAMMPSTest {
protected:
    void SetUp() override
    {
        testbinary = "NeighborListsBin";
        LAMMPSTest::SetUp();
    }

    virtual void create_system(const std::string &atom_style, const std::string &units,
                               const std::string &newton)
    {
        BEGIN_HIDE_OUTPUT();
        command("units " + units);
        command("atom_style " + atom_style);
        command("newton " + newton);
        command("lattice sc 2.0 origin 0.125 0.125 0.125");
        command("region box block -2 2 -2 2 -2 2");
        if ((atom_style == "molecular") || (atom_style == "full")) {
            command(
                "create_box 2 box bond/types 1 extra/bond/per/atom 10 extra/special/per/atom 50");
        } else {
            command("create_box 2 box");
        }
        command("create_atoms 1 box");
        command("pair_style zero 3.5");
        command("pair_coeff * *");
        command("mass * 1.0");
        command("set region box type/ratio 2 0.25 32187");
        command("group one type 1");
        command("group two type 2");
        if ((atom_style == "molecular") || (atom_style == "full")) {
            // add some bonds so there are special bond exclusions
            command("bond_style zero");
            command("bond_coeff 1 2.0");
            command("create_bonds many two one 1 1.9 2.1");
        }
        if ((atom_style == "charge") || (atom_style == "full")) {
            // assign charges
            command("set group one charge 0.25");
            command("set group two charge -1.0");
        }
        END_HIDE_OUTPUT();
    }

    std::vector<std::string> get_neigh_info(const std::string &result)
    {
        if (verbose) utils::print(result);
        auto begin = result.find("Neighbor list info");
        auto end   = result.rfind("Per MPI rank");
        if ((begin != std::string::npos) && (end != std::string::npos) && (begin < end)) {
            return utils::split_lines(result.substr(begin, end - begin));
        } else {
            return std::vector<std::string>();
        }
    }

    int find_first_line(const std::vector<std::string> &list)
    {
        int idx = 0;
        for (const auto &line : list) {
            if (utils::strmatch(line, "neighbor lists")) return idx;
            ++idx;
        }
        return 0;
    }
};

class NeighborListsNsq : public NeighborListsBin {
protected:
    void SetUp() override
    {
        testbinary = "NeighborListsNsq";
        LAMMPSTest::SetUp();
    }

    void create_system(const std::string &atom_style, const std::string &units,
                       const std::string &newton) override
    {
        NeighborListsBin::create_system(atom_style, units, newton);
        BEGIN_HIDE_OUTPUT();
        command("neighbor 2.0 nsq");
        END_HIDE_OUTPUT();
    }
};

class NeighborListsMulti : public NeighborListsBin {
protected:
    void SetUp() override
    {
        testbinary = "NeighborListsMulti";
        LAMMPSTest::SetUp();
    }

    void create_system(const std::string &atom_style, const std::string &units,
                       const std::string &newton) override
    {
        NeighborListsBin::create_system(atom_style, units, newton);
        BEGIN_HIDE_OUTPUT();
        command("neighbor 2.0 multi");
        END_HIDE_OUTPUT();
    }
};

TEST_F(NeighborListsBin, one_atomic_half_list_newton)
{
    create_system("atomic", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/atomonly/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: half/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 26);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 29);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_atomic_half_list_nonewton)
{
    create_system("atomic", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/atomonly/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 80);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 79);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_atomic_half_list_newton_respa)
{
    create_system("atomic", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run_style respa 3 2 2 inner 1 1.0 1.5 middle 2 2.0 2.5 outer 3");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++],
                    ContainsRegex("attributes: half, newton on, respa outer/middle/inner$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/respa/bin/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: half/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 26);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 29);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_atomic_half_list_nonewton_respa)
{
    create_system("atomic", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run_style respa 3 2 2 inner 1 1.0 1.5 middle 2 2.0 2.5 outer 3");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++],
                    ContainsRegex("attributes: half, newton off, respa outer/middle/inner$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/respa/bin/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 80);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 79);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_atomic_half_list_newton_exclude)
{
    create_system("atomic", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("neigh_modify exclude group one one");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/atomonly/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: half/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 5);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 6);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_atomic_half_list_nonewton_exclude)
{
    create_system("atomic", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("neigh_modify exclude group one one");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/atomonly/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 16);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 17);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_hybrid_half_list_newton)
{
    create_system("charge", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style hybrid/overlay lj/cut 3.5 coul/cut 3.5");
    command("pair_coeff * * lj/cut 0.01 2.0");
    command("pair_coeff * * coul/cut");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 17) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("2 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 2 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/atomonly/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: half/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        EXPECT_THAT(neigh_info[lidx++],
                    ContainsRegex(".2. pair coul/cut, perpetual, copy from .1."));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: copy"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 26);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 29);
        nlidx = lammps_find_pair_neighlist(lmp, "coul/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 1);
        inum = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 26);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 29);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_hybrid_half_list_nonewton)
{
    create_system("charge", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style hybrid/overlay lj/cut 3.5 coul/cut 3.5");
    command("pair_coeff * * lj/cut 0.01 2.0");
    command("pair_coeff * * coul/cut");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 17) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("2 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 2 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/atomonly/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        EXPECT_THAT(neigh_info[lidx++],
                    ContainsRegex(".2. pair coul/cut, perpetual, copy from .1.$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: copy"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 80);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 79);
        nlidx = lammps_find_pair_neighlist(lmp, "coul/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 1);
        inum = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 80);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 79);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

#if 0
// FIXME: currently trim is not detected and this test will thus fail
TEST_F(NeighborListsBin, one_trim_half_list_newton)
{
    create_system("charge", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style hybrid/overlay lj/cut 3.5 coul/cut 2.5");
    command("pair_coeff * * lj/cut 0.01 2.0");
    command("pair_coeff * * coul/cut 2.5");
    command("pair_modify neigh/trim yes");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 17) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("2 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 2 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/atomonly/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: half/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".2. pair coul/cut, perpetual, trim from .1."));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on, cut 4.5$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: trim"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 26);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 29);
        nlidx = lammps_find_pair_neighlist(lmp, "coul/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 1);
        inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 19);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 22);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

// FIXME: currently trim is not detected and this test will thus fail
TEST_F(NeighborListsBin, one_trim_half_list_nonewton)
{
    create_system("charge", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style hybrid/overlay lj/cut 3.5 coul/cut 2.5");
    command("pair_coeff * * lj/cut 0.01 2.0");
    command("pair_coeff * * coul/cut 2.5");
    command("pair_modify neigh/trim yes");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 17) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("2 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 2 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/atomonly/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".2. pair coul/cut, perpetual, trim from .1.$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off, cut 4.5$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: trim"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 80);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 79);
        nlidx = lammps_find_pair_neighlist(lmp, "coul/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 1);
        inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 56);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 55);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}
#endif

TEST_F(NeighborListsBin, one_atomic_full)
{
    if (!lammps_config_has_package("MANYBODY")) GTEST_SKIP() << "Missing MANYBODY package for test";
    create_system("atomic", "metal", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style sw");
    command("pair_coeff * * Si.sw Si Si");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair sw, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: full, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: full/bin/atomonly"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "sw", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 92);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 92);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_hybrid_full)
{
    if (!lammps_config_has_package("MANYBODY")) GTEST_SKIP() << "Missing MANYBODY package for test";
    create_system("atomic", "metal", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style hybrid sw lj/cut 3.5");
    command("pair_coeff * * sw Si.sw Si NULL");
    command("pair_coeff 1*2 2 lj/cut 0.01 2.0");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 27) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("4 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 4 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair sw, perpetual, skip from .4.$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: full, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: skip"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        EXPECT_THAT(neigh_info[lidx++],
                    ContainsRegex(".2. pair lj/cut, perpetual, skip from .3.$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: skip"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        EXPECT_THAT(neigh_info[lidx++],
                    ContainsRegex(".3. neighbor class addition, perpetual, half/full from .4.$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: halffull/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".4. neighbor class addition, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: full, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: full/bin/atomonly"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "sw", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int numone = (int)lammps_eval(lmp, "count(one)");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(numone, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 68);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 71);
        nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 1);
        inum = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 8);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 9);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_atomic_full_ghost)
{
    if (!lammps_config_has_package("MANYBODY")) GTEST_SKIP() << "Missing MANYBODY package for test";
    create_system("atomic", "metal", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("neigh_modify one 5000 page 200000");
    command("pair_style airebo 3.0 1 1");
    command("pair_coeff * * CH.airebo C H");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair airebo, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: full, newton on, ghost$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: full/bin/ghost"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/ghost/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "airebo", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 948);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 948);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, two_atomic_half_full)
{
    if (!lammps_config_has_package("MEAM")) GTEST_SKIP() << "Missing MEAM package for test";
    create_system("atomic", "metal", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style meam");
    command("pair_coeff * * library.meam Al Mg NULL Al Mg");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 17) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("2 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 2 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair meam, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: full, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: full/bin/atomonly"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        EXPECT_THAT(neigh_info[lidx++],
                    ContainsRegex(".2. pair meam, perpetual, half/full from .1.$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: halffull/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "meam", 1, 0, 1);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 122);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 122);
        nlidx = lammps_find_pair_neighlist(lmp, "meam", 1, 0, 2);
        EXPECT_EQ(nlidx, 1);
        inum = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 61);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 61);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_molecular_half_list_newton)
{
    if (!lammps_config_has_package("MOLECULE")) GTEST_SKIP() << "Missing MOLECULE package for test";
    create_system("molecular", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: half/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 26);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 23);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_molecular_half_list_nonewton)
{
    if (!lammps_config_has_package("MOLECULE")) GTEST_SKIP() << "Missing MOLECULE package for test";
    create_system("molecular", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 80);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 62);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_molecular_half_list_newton_respa)
{
    if (!lammps_config_has_package("MOLECULE")) GTEST_SKIP() << "Missing MOLECULE package for test";
    create_system("molecular", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run_style respa 2 2 bond 1 pair 2");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: half/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 26);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 23);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsBin, one_molecular_half_list_nonewton_respa)
{
    if (!lammps_config_has_package("MOLECULE")) GTEST_SKIP() << "Missing MOLECULE package for test";
    create_system("molecular", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run_style respa 2 2 bond 1 pair 2");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/bin/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/bin/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: standard"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 80);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 62);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, one_atomic_half_list_newton)
{
    create_system("atomic", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 11) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/nsq/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 46);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 53);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, one_atomic_half_list_nonewton)
{
    create_system("atomic", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 11) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/nsq/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 80);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 79);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, one_atomic_half_list_newton_respa)
{
    create_system("atomic", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run_style respa 3 2 2 inner 1 1.0 1.5 middle 2 2.0 2.5 outer 3");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++],
                    ContainsRegex("attributes: half, newton on, respa outer/middle/inner$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/respa/nsq/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 46);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 53);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, one_atomic_half_list_nonewton_respa)
{
    create_system("atomic", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run_style respa 3 2 2 inner 1 1.0 1.5 middle 2 2.0 2.5 outer 3");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++],
                    ContainsRegex("attributes: half, newton off, respa outer/middle/inner$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/respa/nsq/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 80);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 79);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, one_atomic_half_list_newton_exclude)
{
    create_system("atomic", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("neigh_modify exclude group one one");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 11) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/nsq/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 7);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 10);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, one_atomic_half_list_nonewton_exclude)
{
    create_system("atomic", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("neigh_modify exclude group one one");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 11) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/nsq/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 16);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 17);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, one_atomic_full)
{
    create_system("atomic", "metal", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style sw");
    command("pair_coeff * * Si.sw Si Si");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair sw, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: full, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: full/nsq"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "sw", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 92);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 92);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, two_atomic_half_full)
{
    create_system("atomic", "metal", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style meam");
    command("pair_coeff * * library.meam Al Mg NULL Al Mg");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 16) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("2 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 2 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair meam, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: full, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: full/nsq"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        EXPECT_THAT(neigh_info[lidx++],
                    ContainsRegex(".2. pair meam, perpetual, half/full from .1.$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: halffull/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "meam", 1, 0, 1);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 122);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 122);
        nlidx = lammps_find_pair_neighlist(lmp, "meam", 1, 0, 2);
        EXPECT_EQ(nlidx, 1);
        inum = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 61);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 61);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, one_atomic_full_ghost)
{
    if (!lammps_config_has_package("MANYBODY")) GTEST_SKIP() << "Missing MANYBODY package for test";
    create_system("atomic", "metal", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("neigh_modify one 5000 page 200000");
    command("pair_style airebo 3.0 1 1");
    command("pair_coeff * * CH.airebo C H");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair airebo, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: full, newton on, ghost$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: full/nsq/ghost"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "airebo", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 948);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 948);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, one_molecular_half_list_newton)
{
    if (!lammps_config_has_package("MOLECULE")) GTEST_SKIP() << "Missing MOLECULE package for test";
    create_system("molecular", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/nsq/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 46);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 42);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, one_molecular_half_list_nonewton)
{
    if (!lammps_config_has_package("MOLECULE")) GTEST_SKIP() << "Missing MOLECULE package for test";
    create_system("molecular", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/nsq/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 80);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 62);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}
TEST_F(NeighborListsNsq, one_molecular_half_list_newton_respa)
{
    if (!lammps_config_has_package("MOLECULE")) GTEST_SKIP() << "Missing MOLECULE package for test";
    create_system("molecular", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run_style respa 2 2 bond 1 pair 2");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/nsq/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 46);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 42);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsNsq, one_molecular_half_list_nonewton_respa)
{
    if (!lammps_config_has_package("MOLECULE")) GTEST_SKIP() << "Missing MOLECULE package for test";
    create_system("molecular", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff * * 0.01 2.0");
    command("run_style respa 2 2 bond 1 pair 2");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/nsq/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: none"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: none"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 80);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 62);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}
TEST_F(NeighborListsMulti, one_atomic_half_list_newton)
{
    create_system("atomic", "real", "on");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 4.0");
    command("pair_coeff 1 1 0.01 2.0 2.0");
    command("pair_coeff 1 2 0.01 2.0");
    command("pair_coeff 2 2 0.01 2.0 4.0");
    command("neigh_modify collection/interval 2 2.0 4.0");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton on$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/multi/atomonly/newton"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: half/multi/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: multi"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 50);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 48);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}

TEST_F(NeighborListsMulti, one_atomic_half_list_nonewton)
{
    create_system("atomic", "real", "off");
    BEGIN_CAPTURE_OUTPUT();
    command("pair_style lj/cut 3.5");
    command("pair_coeff 1 1 0.01 2.0 2.0");
    command("pair_coeff 1 2 0.01 2.0");
    command("pair_coeff 2 2 0.01 2.0 4.0");
    command("neigh_modify collection/interval 2 2.0 4.0");
    command("run 0 post no");
    auto neigh_info = get_neigh_info(END_CAPTURE_OUTPUT());
    if (neigh_info.size() >= 12) {
        auto lidx = find_first_line(neigh_info);
        EXPECT_THAT(neigh_info[lidx], ContainsRegex("1 neighbor lists"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("perpetual/occasional/extra = 1 0 0"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex(".1. pair lj/cut, perpetual$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("attributes: half, newton off$"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("pair build: half/multi/atomonly/newtoff"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("stencil: full/multi/3d"));
        EXPECT_THAT(neigh_info[lidx++], ContainsRegex("bin: multi"));
        int nlidx = lammps_find_pair_neighlist(lmp, "lj/cut", 1, 0, 0);
        EXPECT_EQ(nlidx, 0);
        int nlocal = lammps_extract_setting(lmp, "nlocal");
        int inum   = lammps_neighlist_num_elements(lmp, nlidx);
        EXPECT_EQ(nlocal, inum);
        int numneigh   = -1;
        int iatom      = -1;
        int *neighbors = nullptr;
        lammps_neighlist_element_neighbors(lmp, nlidx, 0, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 0);
        EXPECT_EQ(numneigh, 42);
        lammps_neighlist_element_neighbors(lmp, nlidx, 1, &iatom, &numneigh, &neighbors);
        EXPECT_EQ(iatom, 1);
        EXPECT_EQ(numneigh, 43);
    } else {
        GTEST_FAIL() << "No suitable neighbor list info found";
    }
}
} // namespace LAMMPS_NS

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleMock(&argc, argv);

    // handle arguments passed via environment variable
    if (const char *var = getenv("TEST_ARGS")) {
        std::vector<std::string> env = LAMMPS_NS::utils::split_words(var);
        for (auto arg : env) {
            if (arg == "-v") {
                verbose = true;
            }
        }
    }

    if ((argc > 1) && (strcmp(argv[1], "-v") == 0)) verbose = true;

    int rv = RUN_ALL_TESTS();
    MPI_Finalize();
    return rv;
}
