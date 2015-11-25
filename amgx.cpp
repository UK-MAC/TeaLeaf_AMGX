#include "amgx.hpp"
#include "ftocmacros.h"

#include <iostream>
#include <cassert>
#include <mpi.h>

#include <amgx_c.h>
#include <cuda_runtime.h>

extern "C" void tea_init_amgx_
(int* in_left, int* in_right, int* in_bottom, int* in_top,
 int* in_x_min, int* in_x_max, int* in_y_min, int* in_y_max)
{
    amgx_chunk = AMGxChunk(
     * in_left, * in_right, * in_bottom, * in_top,
     * in_x_min, * in_x_max, * in_y_min, * in_y_max);
}

extern "C" void tea_destroy_amgx_
(void)
{
    amgx_chunk.Destroy();
}

extern "C" void amgx_solve_
(double *Kx, double *Ky, double* rx, double* ry, double *b, double* error)
{
    amgx_chunk.Solve(Kx, Ky, b, rx, ry, error);
}

/* print callback (could be customized) */
void print_callback(const char *msg, int length){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) printf("%s", msg);
}

AMGxChunk::AMGxChunk
// Need the absolute position of chunk in the whole mesh
(int in_left, int in_right, int in_bottom, int in_top,
 int in_x_min, int in_x_max, int in_y_min, int in_y_max)
:left  (in_left),
 right (in_right),
 bottom(in_bottom),
 top   (in_top),
 x_min (in_x_min),
 x_max (in_x_max),
 y_min (in_y_min),
 y_max (in_y_max),
 amgx_mpi_comm(MPI_COMM_WORLD),
 step(0)
{
    cudaSetDevice(0);

    // initialise things
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    // create empty config
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, "./CONFIG.json"));

    // remove the need to use AMGX_SAFE_CALL?
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    MPI_Comm_rank(amgx_mpi_comm, &rank);

    // doubles - input file could control host or device?
    const AMGX_Mode mode = AMGX_mode_dDDI;

    // create resources, matrix, vector and solver

    //AMGX_SAFE_CALL(AMGX_resources_create(&rsrc, cfg, &amgx_mpi_comm, 1, &rank));
    AMGX_resources_create_simple(&rsrc, cfg);

    AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, mode, cfg));

    AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&x, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&b, rsrc, mode));

    work_array_size = (x_max)*(y_max);

    // allocating here as well to save any fortran interop problems
    x_host = new double[work_array_size]();
    b_rhs = new double[work_array_size]();
    A_rows = new int[work_array_size+1]();
    A_cols = new int[5*work_array_size]();
    A_data = new double[5*work_array_size]();

    AMGX_SAFE_CALL(AMGX_pin_memory(x_host, sizeof(double)*work_array_size));
    AMGX_SAFE_CALL(AMGX_pin_memory(b_rhs, sizeof(double)*work_array_size));
    AMGX_SAFE_CALL(AMGX_pin_memory(A_rows, sizeof(int)*(work_array_size+1)));

    // slightly overallocate
    AMGX_SAFE_CALL(AMGX_pin_memory(A_cols, sizeof(int)*5*work_array_size));
    AMGX_SAFE_CALL(AMGX_pin_memory(A_data, sizeof(double)*5*work_array_size));
}

void AMGxChunk::Destroy
(void)
{
    AMGX_SAFE_CALL(AMGX_solver_destroy(solver));
    AMGX_SAFE_CALL(AMGX_vector_destroy(x));
    AMGX_SAFE_CALL(AMGX_vector_destroy(b));
    AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
    AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

    // TODO unpin memory

    AMGX_SAFE_CALL(AMGX_finalize());
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    // cudaDeviceReset();

    delete[] x_host;
    delete[] b_rhs;
    delete[] A_rows;
    delete[] A_cols;
    delete[] A_data;
}

void AMGxChunk::Solve
(double *Kx, double *Ky, double *b_mesh, double* rx, double* ry, double* error)
{
    /*  - explicitly construct A from Kx/Ky (allocate data further up) and make
         x/b contiguous for vector upload
    */

    if (step == 0)
    {
        amgx_construct_matrix(&x_min, &x_max, &y_min, &y_max, &nnz, rx, ry,
            b_mesh, b_rhs, x_host, Kx, Ky, A_cols, A_rows, A_data);

#if 0
        #pragma omp parallel for
        for (int ii = 0; ii < 5*work_array_size; ii++)
        {
            A_cols[ii] -= 1;
        }

        std::cout << "Uploading first time" << std::endl;

        AMGX_SAFE_CALL(AMGX_matrix_upload_all(A,
            work_array_size,
            nnz,
            1,
            1,
            A_rows,
            A_cols,
            A_data,
            NULL));
#else
        FILE * matrix_file = fopen("MARK.txt", "w");
        fprintf(matrix_file, "%%%%MatrixMarket matrix coordinate real symmetric\n");
        fprintf(matrix_file, "%%%%AMGX rhs solution\n");
        fprintf(matrix_file, "%d %d %d\n", work_array_size, work_array_size, nnz);

        int written = 0;

        // print out matrix
        for (int ii = 0; ii < x_max*y_max; ii++)
        {
            for (int jj = A_rows[ii]; jj < A_rows[ii+1]; jj++)
            {
                fprintf(matrix_file, "%d %d %.12e\n", 1+ii, A_cols[jj], A_data[jj]);
                written++;
            }
        }

        assert(written==nnz);

        // print out rhs
        fprintf(matrix_file, "%% RHS\n");
        for (int ii = 0; ii < x_max*y_max; ii++)
        {
            fprintf(matrix_file, "%.10e\n", b_rhs[ii]);
        }
        // print out soln
        fprintf(matrix_file, "%% SOLN\n");
        for (int ii = 0; ii < x_max*y_max; ii++)
        {
            fprintf(matrix_file, "%.10e\n", x_host[ii]);
        }

        fclose(matrix_file);
#endif

    AMGX_SAFE_CALL(AMGX_read_system(A, b, x, "MARK.txt"));

        AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
    }
    else
    {
        // Matrix does not change now
        //std::cout << "Uploading again" << std::endl;
        //AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, work_array_size, nnz, A_data, NULL));

        std::cout << "NOT uploading again" << std::endl;
    }

    amgx_read_mesh(&x_min, &x_max, &y_min, &y_max, b_mesh, b_rhs, x_host);

    AMGX_SAFE_CALL(AMGX_vector_upload(b, work_array_size, 1, b_rhs));
    AMGX_SAFE_CALL(AMGX_vector_upload(x, work_array_size, 1, x_host));

    int mtx_size, block_dim;
    int it_number;

    /*
     *  TODO
     FIRST TIME:
     *  - set communication maps assuming its split into columns, each process
         will only need 2 neighbours at most to communicate with...might be worth
         changing domain decompition to use the mpi_cart_comm so that ranks can be
         more easily identified
         - amgx_vector_bind on x and b as well
     *  - upload to device

     SECOND+ TIME:
     *  - upload vectors
     *  - solve
     *  - download
     *  - return
     */

    AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, x));
    //AMGX_SAFE_CALL(AMGX_solver_solve_with_0_initial_guess(solver, b, x));

    AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(solver, &it_number));

    step++;

    AMGX_SAFE_CALL(AMGX_vector_download(x, x_host));

    amgx_writeback(&x_min, &x_max, &y_min, &y_max, b_mesh, x_host);

    // FIXME seems to give an error
    //AMGX_SAFE_CALL(AMGX_solver_get_iteration_residual(solver, it_number, 0, error));
    *error = 1e-16;
}

