#include <mpi.h>
#include <amgx_c.h>

class AMGxChunk {
private:
    int left, right, bottom, top;
    int x_min, x_max, y_min, y_max;

    int * A_rows;
    int * A_cols;
    double * A_data;
    double * b_rhs;
    double * x_host;

    // dimension of matrix
    int work_array_size;
    // nnz in matrix
    int nnz;

    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    //status handling
    AMGX_SOLVE_STATUS status;

    MPI_Comm amgx_mpi_comm;
    int rank;

    // which step - only upload to device on first step
    int step;
public:
    void Solve
    (double *Kx, double *Ky, double *b, double* rx, double* ry, double* error);

    void Destroy
    (void);

    AMGxChunk
    (void){}

    AMGxChunk
    // Need the absolute position of chunk in the whole mesh
    (int left, int right, int bottom, int top,
     int x_min, int x_max, int y_min, int y_max);
};

AMGxChunk amgx_chunk;

extern "C" void amgx_construct_matrix
(int*, int*, int*, int*, int*, double*, double*,
    double*, double*, double*, double*, double*, int*, int*, double*);

extern "C" void amgx_read_mesh
(int*, int*, int*, int*, double*, double*, double*);

extern "C" void amgx_writeback
(int*, int*, int*, int*, double*, double*);

