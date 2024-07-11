////////////////////////////////////////////////////////////////////////////////
// TEST ENVIRONMENT
// OS: Ubuntu 22.04
// mfem: HEAD detached at v4.6
//       build config: MFEM_USE_OPENMP=YES MFEM_USE_MPI=YES MFEM_DEBUG=NO
// hypre: HEAD detached at v2.24.0
//        build config: --with-openmp
// metis: 4.0.3 (symlinked as metis-4.0)
//        build options: defaults
// cmake version 3.29.0
// g++ (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
////////////////////////////////////////////////////////////////////////////////
// STEPS TO REPRODUCE
// 1. cd examples
// 2. make repro
// 3. ./repro
////////////////////////////////////////////////////////////////////////////////

#include <array>
#include <iostream>

#include <mfem.hpp>

auto expect_no_hypre_error() -> void {
    HYPRE_Int error_code = HYPRE_GetError();
    if (error_code != 0) {
        auto error_descr = std::array<char, 1024>();
        HYPRE_DescribeError(error_code, error_descr.data());
        std::cerr << "Hypre error code: " << error_code << ", " << error_descr.data() << std::endl;
        throw std::runtime_error("Hypre error occurred.");
    }
}

auto filter_density_field() -> void {
    // mesh has 1 physical volume: label 1, and 6 physical surfaces, labels 1..6
    auto mesh = mfem::Mesh::MakeCartesian3D(10, 10, 10, mfem::Element::TETRAHEDRON, 10.0, 10.0, 10.0);

    auto pmesh = mfem::ParMesh(MPI_COMM_WORLD, mesh);

    auto serial_elem_fec = mfem::L2_FECollection(0, pmesh.SpaceDimension());
    auto serial_elem_fes = mfem::FiniteElementSpace(&mesh, &serial_elem_fec, 1, mfem::Ordering::byVDIM);
    auto elem_fes = mfem::ParFiniteElementSpace(serial_elem_fes, pmesh);

    auto serial_node_fec = mfem::H1_FECollection(1, pmesh.SpaceDimension());
    auto serial_node_fes = mfem::FiniteElementSpace(&mesh, &serial_node_fec, 1, mfem::Ordering::byNODES);
    auto node_fes = mfem::ParFiniteElementSpace(serial_node_fes, pmesh);

    auto inv_vol = mfem::Vector(elem_fes.GetTrueVSize());
    for (int i = 0; i < inv_vol.Size(); i++) {
        inv_vol[i] = 1.0 / pmesh.GetElementVolume(i);
    }

    auto radius = 3.0;
    auto r = radius / (2.0 * sqrt(3.0));
    auto r_coeff = mfem::ConstantCoefficient(r * r);
    auto mass_coeff = mfem::ConstantCoefficient(1.0);
    auto inv_vol_gf = mfem::ParGridFunction(&elem_fes);
    inv_vol_gf.SetFromTrueDofs(inv_vol);

    mfem::ParBilinearForm k_form(&node_fes);
    k_form.AddDomainIntegrator(new mfem::DiffusionIntegrator(r_coeff));
    k_form.AddDomainIntegrator(new mfem::MassIntegrator(mass_coeff));
    k_form.Assemble();
    //k_form.Finalize();
    auto ess_tdof_list = mfem::Array<int>();
    auto ess_bdr = mfem::Array<int>(pmesh.bdr_attributes.Max());
    ess_bdr = 0;
    k_form.FESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    auto a_matrix = mfem::HypreParMatrix();
    k_form.FormSystemMatrix(ess_tdof_list, a_matrix);

    mfem::ParMixedBilinearForm t_form(&elem_fes, &node_fes);
    t_form.AddDomainIntegrator(new mfem::MixedScalarMassIntegrator(mass_coeff));
    t_form.Assemble();
    t_form.Finalize();
    auto t_matrix = std::unique_ptr<mfem::HypreParMatrix const>(t_form.ParallelAssemble());

    auto inv_vol_coeff = mfem::GridFunctionCoefficient(&inv_vol_gf);
    mfem::ParMixedBilinearForm t_star_form(&node_fes, &elem_fes);
    t_star_form.AddDomainIntegrator(new mfem::MixedScalarMassIntegrator(inv_vol_coeff));
    t_star_form.Assemble();
    t_star_form.Finalize();
    auto t_star_matrix = std::unique_ptr<mfem::HypreParMatrix const>(t_star_form.ParallelAssemble());

    auto elem_density_gf = mfem::ParGridFunction(&elem_fes);
    elem_density_gf = 1.0;

    auto node_density = mfem::Vector(node_fes.GetTrueVSize());
    t_matrix->Mult(elem_density_gf, node_density);
    expect_no_hypre_error();

    auto node_density_filtered = mfem::Vector(node_fes.GetTrueVSize());
    node_density_filtered = 0.0;

    auto amg = mfem::HypreBoomerAMG(a_matrix);
    amg.SetPrintLevel(2);
    auto pcg = mfem::HyprePCG(a_matrix);
    pcg.SetTol(1e-12);
    pcg.SetMaxIter(10000);
    pcg.SetPrintLevel(2);
    pcg.SetPreconditioner(amg);
    pcg.Mult(node_density, node_density_filtered);
    expect_no_hypre_error();

    auto elem_density_filtered_gf = mfem::ParGridFunction(&elem_fes);
    elem_density_filtered_gf = 0.0;
    t_star_matrix->Mult(node_density_filtered, elem_density_filtered_gf);
    expect_no_hypre_error();
}

int main(int argc, char* argv[]) {
    mfem::Mpi::Init();
    mfem::Hypre::Init();

    filter_density_field();
}
