#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

#define WORKTAG 1
#define DIETAG 2

#define PI 3.141592653589793238463

using namespace std;

int main(int argc, char *argv[])
{
    int	nrank, ntasks;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    int rank;
    MPI_Status status;

    int N = 16740, n = 2 * 16740 / (ntasks - 1);
    double buf[n], result = 0.0;

    double exact_solution = (4 * PI) / 3, V = 8.0, cur_solution = 0.0;
    double cur_solution_result;
    double eps = atof(argv[1]);

    int cum_n = 0;

    double matrix[n];

    double t1, t, Time;

    srand(1);

    t1 = MPI_Wtime();

    do
    {
        if(nrank == 0)
        {
            for (rank = 1; rank < ntasks; ++rank)
            {
                for(int i = 0; i < n; i = i + 2)
                {
                    matrix[i] = -1.0 + (double)rand() / RAND_MAX * (1.0 - (-1.0));
                    matrix[i + 1] = -1.0 + (double)rand() / RAND_MAX * (1.0 - (-1.0));
                }

                MPI_Send(matrix, n, MPI_DOUBLE, rank, WORKTAG, MPI_COMM_WORLD);
            }
        }

        if(nrank != 0)
        {
            MPI_Recv(buf, n, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if(status.MPI_TAG == DIETAG)
            {
                break;
            }

            result = 0.0;

            for(int i = 0; i < n; i = i + 2)
            {
                if(buf[i] * buf[i] + buf[i + 1] * buf[i + 1] <= 1.0)
                {
                    result += sqrt(buf[i] * buf[i] + buf[i + 1] * buf[i + 1]) * V;
                }
            }
        }

        MPI_Reduce(&result, &cur_solution_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if(nrank == 0)
        {
            cum_n += N;
            cur_solution += cur_solution_result;
            if(fabs(cur_solution / cum_n - exact_solution) < eps)
            {
                for(rank = 1; rank < ntasks; ++rank)
                {
                    MPI_Send(0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
                }
                break;
            }
        }
    }
    while(true);

    t = MPI_Wtime() - t1;

    MPI_Reduce(&t, &Time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(nrank == 0)
    {
        cout << "Result:             " << cur_solution / cum_n << endl;
        cout << "Error:              " << fabs(exact_solution - cur_solution / cum_n) << endl;
        cout << "Number of points:   " << cum_n << endl;
        cout << "Time:               " << Time << " sec" << endl;
    }

    MPI_Finalize();
}
