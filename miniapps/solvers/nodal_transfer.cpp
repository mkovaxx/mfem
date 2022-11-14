#include <mfem.hpp>
#include <fstream>
#include <iostream>
#include "../common/mfem-common.hpp"
#include <cmath>


/// Generate second order mesh on 4 processes
/// mpirun -np 4 ./nodal_transfer -rs 2 -rp 1 -gd 1 -o 2
/// Read the generated data and map it to a grid function
/// defined on two processes
/// mpirun -np 2 ./nodal_transfer -rs 2 -rp 0 -gd 0 -snp 4 -o 2
///
/// Generate first order grid function on 8 processes
/// mpirun -np 8 ./nodal_transfer -rs 2 -rp 2 -gd 1 -o 1 -m ../../data/star.mesh
/// Read the generated data on 4 processes and coarser mesh
/// mpirun -np 4 ./nodal_transfer -rs 2 -rp 0 -gd 0 -snp 8 -o 1 -m ../../data/star.mesh
///



using namespace mfem;

class TestCoeff:public Coefficient
{
public:
   TestCoeff()
   {

   }

   virtual
   double Eval(ElementTransformation &T,
               const IntegrationPoint &ip)
   {
      if (T.GetSpaceDim()==3)
      {
         double x[3];
         Vector transip(x, 3);
         T.Transform(ip, transip);
         return std::sin(x[0])*std::cos(x[1]) +
                std::sin(x[1])*std::cos(x[2]) +
                std::sin(x[2])*std::cos(x[0]);
      }
      else
      {
         double x[2];
         Vector transip(x, 2);
         T.Transform(ip, transip);
         return std::sin(x[0])*std::cos(x[1]) +
                std::sin(x[1])*std::cos(x[0]);
      }
   }
};


int main(int argc, char* argv[])
{
   // Initialize MPI.
   Mpi::Init(argc, argv);
   int myrank = Mpi::WorldRank();

   // 2. Parse command-line options
   const char *mesh_file = "../../data/beam-tet.mesh";
   int ser_ref_levels = 3;
   int par_ref_levels = 1;
   int order = 1;
   int gen_data = 1;
   int src_num_procs = 4;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref_levels,
                  "-rs",
                  "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels,
                  "-rp",
                  "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&gen_data,
                  "-gd",
                  "--generate-data",
                  "Generate input data for the transfer.");
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&src_num_procs,
                  "-snp",
                  "--src_num_procs",
                  "Number of processes for the src grid function.");
   args.Parse();
   if (!args.Good())
   {
      if (myrank == 0)
      {
         args.PrintUsage(std::cout);
      }
      return 1;
   }

   if (myrank == 0)
   {
      args.PrintOptions(std::cout);
   }

   //    Read the (serial) mesh from the given mesh file on all processors. We
   //    can handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->SpaceDimension();

   //    Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   //    Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   //   Define the finite element spaces for the solution
   H1_FECollection fec(order, dim);
   //L2_FECollection fec(order,dim);
   ParFiniteElementSpace fespace(pmesh, &fec, 2, Ordering::byVDIM);
   //ParFiniteElementSpace fespace(pmesh, &fec, 2, Ordering::byNODES);
   HYPRE_Int glob_size = fespace.GlobalTrueVSize();
   if (myrank == 0)
   {
      std::cout << "Number of finite element unknowns: " << glob_size
                << std::endl;
   }

   if (gen_data)
   {
      Vector coo; // coordinates associated with fespace nodes
      coo.SetSize(fespace.GetVSize()*dim/(fespace.GetVDim()));
      {
         ElementTransformation *trans;
         const IntegrationRule* ir=nullptr;
         Array<int> vdofs;
         DenseMatrix elco;
         int isca=1;
         if (fespace.GetOrdering()==Ordering::byVDIM) {isca=fespace.GetVDim();}

         for (int i=0; i<fespace.GetNE(); i++)
         {
            const FiniteElement* el=fespace.GetFE(i);
            //get the element transformation
            trans = fespace.GetElementTransformation(i);
            ir=&(el->GetNodes());
            fespace.GetElementVDofs(i,vdofs);
            elco.SetSize(dim,ir->GetNPoints());
            trans->Transform(*ir,elco);
            for (int p=0; p<ir->GetNPoints(); p++)
            {
               for (int d=0; d<dim; d++)
               {
                  coo[vdofs[p]*dim/isca+d]=elco(d,p);
               }
            }
         }
      }

      ParGridFunction x(&fespace);
      TestCoeff prco;
      Coefficient* coef[2]; coef[0]=&prco; coef[1]=&prco;
      x.ProjectCoefficient(coef);

      //  Save the grid function
      {
         //save the mesh and the data
         std::ostringstream oss;
         oss << std::setw(10) << std::setfill('0') << myrank;
         std::string mname="coords_"+oss.str()+".msh";
         std::string gname="grid_func_"+oss.str()+".gf";
         std::ofstream out;

         // save the points
         out.open(mname.c_str(),std::ios::out);
         out.precision(20);
         out<<coo.Size()<<std::endl;
         coo.Print(out,dim);
         out.close();


         //save the grid function data
         out.open(gname.c_str(),std::ios::out);
         out.precision(20);
         out<<x.Size()<<std::endl;
         x.Print(out);
         out.close();
      }

   }
   else
   {
      // read the grid function written to files
      //  and map it to the current partition scheme

      // x-grid function will be the target of the transfer
      ParGridFunction x(&fespace); x=0.0;
      // y will be utilized later for comparison
      ParGridFunction y(&fespace);

      TestCoeff prco;
      Coefficient* coef[2]; coef[0]=&prco; coef[1]=&prco;
      y.ProjectCoefficient(coef);

      // Map the src grid function
      {
         std::ifstream in;

         KDTreeNodalTransfer map(x);

         Vector sgf;
         Vector coords;

         for (int p=0; p<src_num_procs; p++)
         {
            std::ostringstream oss;
            oss << std::setw(10) << std::setfill('0') << p;
            std::string mname="coords_"+oss.str()+".msh";
            std::string gname="grid_func_"+oss.str()+".gf";

            in.open(mname.c_str(),std::ios::in);
            coords.Load(in);
            in.close();

            in.open(gname.c_str(),std::ios::in);
            sgf.Load(in);
            in.close();

            map.Transfer(coords,sgf,Ordering::byVDIM);
         }

      }

      //write the result into a ParaView file
      {
         ParaViewDataCollection paraview_dc("GridFunc", pmesh);
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(order);
         paraview_dc.SetDataFormat(mfem::VTKFormat::BINARY);
         paraview_dc.SetCycle(0);
         paraview_dc.SetTime(0.0);
         paraview_dc.RegisterField("x",&x);
         paraview_dc.RegisterField("y",&y);
         paraview_dc.Save();
      }

      //compare the results
      Vector tmpv=x;
      tmpv-=y;
      double l2err=mfem::InnerProduct(MPI_COMM_WORLD,tmpv,tmpv);
      if (myrank==0)
      {
         std::cout<<"|l2 error|="<<sqrt(l2err)<<std::endl;
      }

   }

   delete pmesh;
   return 0;
}
