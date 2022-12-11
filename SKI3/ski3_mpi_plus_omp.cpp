#include <iostream>
#include <mpi.h>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <omp.h>

using namespace std;

double k(double x, double y)
{
    return 1.0 + (x + y) * (x + y);
}

double u(double x, double y)
{
    return 1.0 + x * x + y * y;
}

double psi(double x, double y, char c)
{
    double a;

    a = (k(x, y) * 4) / (u(x, y) * u(x, y));

    if(c == 'u')
    {
        return a * (-y);
    }
    else if(c == 'd')
    {
        return a * y;
    }
    else if(c == 'r')
    {
        return a * (-x);
    }
    else if(c == 'l')
    {
        return a * x;
    }
}

double F(double x, double y)
{
    return 2.0 / u(x, y) - (((16.0 * x * x * k(x, y)) / (u(x, y) * u(x, y) * u(x, y)) - (4 * x * (2 * x + 2 * y)) / (u(x, y) * u(x, y)) - (4 * k(x, y)) / (u(x, y) * u(x, y))) + ((16.0 * y * y * k(x, y)) / (u(x, y) * u(x, y) * u(x, y)) - (4 * y * (2 * x + 2 * y)) / (u(x, y) * u(x, y)) - (4 * k(x, y)) / (u(x, y) * u(x, y))));
}

double skalar(double a[], double b[], int m, int n, double h1, double h2)
{
    double sum = 0.0;

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            sum += a[i * n + j] * b[i * n + j];
        }
    }

    return sum;
}

int main(int argc, char *argv[])
{
    int M = atoi(argv[1]), N = atoi(argv[2]);

    double eps = 0.000001, cur_eps;

    double tau;

    double h1 = (3.0 - (-2.0)) / M, h2 = (4.0 - (-1.0)) / N;

    int	nrank, nproc;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Status status;

    int m, n, res_m, res_n;
    int l, q;

    double t1, t, Time;

    t1 = MPI_Wtime();

    if (sqrt((int)nproc) == (int)sqrt((int)nproc))
    {
        m = (M + 1) / (int)sqrt((int)nproc);
        n = (N + 1) / (int)sqrt((int)nproc);
        res_m = (M + 1) % (int)sqrt((int)nproc);
        res_n = (N + 1) % (int)sqrt((int)nproc);

        if(nrank == nproc - 1)
        {
            m = m + res_m;
            n = n + res_n;
        }
        else if((nrank >= nproc - sqrt((int)nproc)) && (nrank != nproc - 1))
        {
            m = m + res_m;
        }
        else if(((nrank + 1) % (int)sqrt((int)nproc) == 0) && (nrank != nproc - 1))
        {
            n = n + res_n;
        }

        l = (int)sqrt((int)nproc);
        q = (int)sqrt((int)nproc);
    }
    else
    {
        for(int i = 1; i * i < (int)nproc; i++)
        {
            if((int)nproc % i == 0)
            {
                l = i;
            }
        }

        q = (int)nproc / l;

        m = (M + 1) / l;
        n = (N + 1) / q;
        res_m = (M + 1) % l;
        res_n = (N + 1) % q;

        if(nrank == nproc - 1)
        {
            if(l != 1)
                m = m + res_m;
            if(res_n == 1)
                n = n + res_n;
            else
                n = n + 1;
        }
        else if((nrank >= nproc - q) && (nrank != nproc - 1))
        {
            if(l != 1)
                m = m + res_m;
            if(q - (nrank + 1) % q < res_n)
                n = n + 1;
        }
        else if(((nrank + 1) % q == 0) && (nrank != nproc - 1))
        {
            if(res_n == 1)
                n = n + res_n;
            else
                n = n + 1;
        }
        else if((q - (nrank + 1) % q < res_n) && (res_n != 1) && ((nrank + 1) % q != 0) && (nrank != nproc - 1))
        {
            n = n + 1;
        }
    }

    double ***setka = new double** [m];
    for(int i = 0; i < m; i++)
    {
        setka[i] = new double* [n];
    }
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            setka[i][j] = new double[2];
        }
    }

    int *sizes = new int[2];
    int *Sizes = new int[2 * nproc];

    sizes[0] = m;
    sizes[1] = n;

    MPI_Allgather(sizes, 2, MPI_INT, Sizes, 2, MPI_INT, MPI_COMM_WORLD);

    int *cum_sizes = new int[q];

    for(int i = 0; i < q; i++)
    {
        cum_sizes[i] = 0;
    }

    for(int i = 1; i < q; i++)
    {
        cum_sizes[i] = cum_sizes[i - 1] + Sizes[2 * (i - 1) + 1];
    }

    double **A = new double* [m * n];
    for(int i = 0; i < m * n; i++)
    {
        A[i] = new double[5];
    }

    double *B = new double[m * n];
    double *w = new double[m * n];
    double *w_last = new double[m * n];
    double *w_up = new double[m];
    double *w_down = new double[m];
    double *w_left = new double[n];
    double *w_right = new double[n];
    double *w_up_0 = new double[m];
    double *w_down_0 = new double[m];
    double *w_left_0 = new double[n];
    double *w_right_0 = new double[n];
    double *r = new double[m * n];
    double *Ar = new double[m * n];

    double *r_0 = new double[(M + 1) * (N + 1)];
    double *Ar_0 = new double[(M + 1) * (N + 1)];

    #pragma omp parallel num_threads(4)
    {
        #pragma omp for
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                if(nrank == 0)
                {
                    setka[i][j][0] = -2 + (i + (nrank / q) * Sizes[2 * nrank]) * h1;
                    setka[i][j][1] = -1 + (j + (nrank % q) * Sizes[2 * nrank + 1]) * h2;
                }
                else
                {
                    if((nrank >= 1) && (nrank <= q - 1))
                    {
                        setka[i][j][0] = -2 + (i + (nrank / q) * Sizes[2 * nrank]) * h1;
                        setka[i][j][1] = -1 + (j + cum_sizes[nrank % q]) * h2;
                    }
                    else if(nrank % q == 0)
                    {
                        setka[i][j][0] = -2 + (i + (nrank / q) * Sizes[2 * (nrank - q)]) * h1;
                        setka[i][j][1] = -1 + (j + (nrank % q) * Sizes[2 * nrank + 1]) * h2;
                    }
                    else
                    {
                        setka[i][j][0] = -2 + (i + (nrank / q) * Sizes[2 * (nrank - q)]) * h1;
                        setka[i][j][1] = -1 + (j + cum_sizes[nrank % q]) * h2;
                    }
                }
            }
        }

        #pragma omp for
        for(int i = 0; i < m * n; i++)
        {
            w[i] = 0.0;
        }

        #pragma omp for
        for(int i = 0; i < m * n; i++)
        {
            #pragma omp collapse(2)
            for(int j = 0; j < 5; j++)
            {
                A[i][j] = 0.0;
            }
        }
    }

    int **pos = new int* [m * n];
    for(int i = 0; i < m * n; i++)
    {
        pos[i] = new int[5];
    }

    int *revcounts = new int[nproc];
    int *displs = new int[nproc];

    int revcount;

    revcount = m * n;

    MPI_Allgather(&revcount, 1, MPI_INT, revcounts, 1, MPI_INT, MPI_COMM_WORLD);

    displs[0] = 0;

    for(int i = 1; i < nproc; i++)
    {
        displs[i] = displs[i - 1] + revcounts[i - 1];
    }

    if(nrank == 0)
    {
        A[0][0] = 1.0 + (2.0 / (h1 * h1)) * k(setka[0][0][0] + 0.5 * h1, setka[0][0][1]) + (2.0 / (h2 * h2)) * k(setka[0][0][0], setka[0][0][1] + 0.5 * h2);
        pos[0][0] = 0;
        A[0][1] = (-2.0 / (h1 * h1)) * k(setka[0][0][0] + 0.5 * h1, setka[0][0][1]);
        pos[0][1] = n;
        A[0][2] = (-2.0 / (h2 * h2)) * k(setka[0][0][0], setka[0][0][1] + 0.5 * h2);
        pos[0][2] = 1;
        pos[0][3] = -5;
        pos[0][4] = -5;
        B[0] = F(setka[0][0][0], setka[0][0][1]) + (2.0 / h1 + 2.0 / h2) * (h1 * psi(setka[0][0][0], setka[0][0][1], 'd') + h2 * psi(setka[0][0][0], setka[0][0][1], 'l')) / (h1 + h2);
    }

    if(nrank == nproc - q)
    {
        A[m * n - n][0] = 1.0 + (2.0 / (h1 * h1)) * k(setka[m-1][0][0] - 0.5 * h1, setka[m-1][0][1]) + (2.0 / (h2 * h2)) * k(setka[m-1][0][0], setka[m-1][0][1] + 0.5 * h2);
        pos[m * n - n][0] = m * n - n;
        A[m * n - n][1] = (-2.0 / (h1 * h1)) * k(setka[m-1][0][0] - 0.5 * h1, setka[m-1][0][1]);
        pos[m * n - n][1] = (m - 2) * n;
        A[m * n - n][2] = (-2.0 / (h2 * h2)) * k(setka[m-1][0][0], setka[m-1][0][1] + 0.5 * h2);
        pos[m * n - n][2] = m * n - n + 1;
        pos[m * n - n][3] = -5;
        pos[m * n - n][4] = -5;
        B[m * n - n] = F(setka[m-1][0][0], setka[m-1][0][1]) + (2.0 / h1 + 2.0 / h2) * (h1 * psi(setka[m-1][0][0], setka[m-1][0][1], 'd') + h2 * psi(setka[m-1][0][0], setka[m-1][0][1], 'r')) / (h1 + h2);
    }

    if(nrank == nproc - 1)
    {
        A[m * n - 1][0] = 1.0 + (2.0 / (h1 * h1)) * k(setka[m-1][n-1][0] - 0.5 * h1, setka[m-1][n-1][1]) + (2.0 / (h2 * h2)) * k(setka[m-1][n-1][0], setka[m-1][n-1][1] - 0.5 * h2);
        pos[m * n - 1][0] = m * n - 1;
        A[m * n - 1][1] = (-2.0 / (h1 * h1)) * k(setka[m-1][n-1][0] - 0.5 * h1, setka[m-1][n-1][1]);
        pos[m * n - 1][1] = (m - 1) * n - 1;
        A[m * n - 1][2] = (-2.0 / (h2 * h2)) * k(setka[m-1][n-1][0], setka[m-1][n-1][1] - 0.5 * h2);
        pos[m * n - 1][2] = m * n - 2;
        B[m * n - 1] = F(setka[m-1][n-1][0], setka[m-1][n-1][1]) + (2.0 / h1 + 2.0 / h2) * (h1 * psi(setka[m-1][n-1][0], setka[m-1][n-1][1], 'u') + h2 * psi(setka[m-1][n-1][0], setka[m-1][n-1][1], 'r')) / (h1 + h2);
        pos[m * n - 1][3] = -5;
        pos[m * n - 1][4] = -5;
    }

    if(nrank == q - 1)
    {
        A[n - 1][0] = 1.0 + (2.0 / (h1 * h1)) * k(setka[0][n-1][0] + 0.5 * h1, setka[0][n-1][1]) + (2.0 / (h2 * h2)) * k(setka[0][n-1][0], setka[0][n-1][1] - 0.5 * h2);
        pos[n - 1][0] = n - 1;
        A[n - 1][1] = (-2.0 / (h1 * h1)) * k(setka[0][n-1][0] + 0.5 * h1, setka[0][n-1][1]);
        pos[n - 1][1] = 2 * (n - 1) + 1;
        A[n - 1][2] = (-2.0 / (h2 * h2)) * k(setka[0][n-1][0], setka[0][n-1][1] - 0.5 * h2);
        pos[n - 1][2] = n - 2;
        B[n - 1] = F(setka[0][n-1][0], setka[0][n-1][1]) + (2.0 / h1 + 2.0 / h2) * (h1 * psi(setka[0][n-1][0], setka[0][n-1][1], 'u') + h2 * psi(setka[0][n-1][0], setka[0][n-1][1], 'l')) / (h1 + h2);
        pos[n - 1][3] = -5;
        pos[n - 1][4] = -5;
    }

    int i_b, j_b, i_e, j_e;
    if(nrank == 0)
    {
        j_b = 1;
        i_b = 1;

        if(nproc == 1)
        {
            j_e = n - 1;
            i_e = m - 1;
        }
        else if(nproc == 2)
        {
            j_e = n;
            i_e = m - 1;
        }
        else
        {
            j_e = n;
            i_e = m;
        }
    }
    else if(nrank == q - 1)
    {
        if(nproc != 2)
        {
            j_b = 0;
            i_b = 1;
            j_e = n - 1;
            i_e = m;
        }
        else
        {
            j_b = 0;
            i_b = 1;
            j_e = n - 1;
            i_e = m - 1;
        }
    }
    else if(nrank == nproc - q)
    {
        j_b = 1;
        i_b = 0;
        j_e = n;
        i_e = m - 1;
    }
    else if(nrank == nproc - 1)
    {
        j_b = 0;
        i_b = 0;
        j_e = n - 1;
        i_e = m - 1;
    }
    else if((nrank >= 1) && (nrank <= q - 2))
    {
        i_e = m;
        i_b = 1;
        j_b = 0;
        j_e = n;
    }
    else if((nrank <= nproc - 2) && (nrank >= nproc - q + 1))
    {
        i_e = m - 1;
        i_b = 0;
        j_b = 0;
        j_e = n;
    }
    else if(nrank % q == 0)
    {
        i_e = m;
        i_b = 0;
        j_b = 1;
        j_e = n;
    }
    else if(nrank % q == q - 1)
    {
        i_e = m;
        i_b = 0;
        j_b = 0;
        j_e = n - 1;
    }
    else
    {
        i_e = m;
        i_b = 0;
        j_b = 0;
        j_e = n;
    }

    #pragma omp parallel num_threads(4)
    {
        if((nrank >= 0) && (nrank <= q - 1))
        {
            #pragma omp for
            for(int j = j_b; j < j_e; j++)
            {
                A[j][0] = 1.0 + (2.0 / (h1 * h1)) * k(setka[0][j][0] + 0.5 * h1, setka[0][j][1]) + (1.0 / (h2 * h2)) * k(setka[0][j][0], setka[0][j][1] - 0.5 * h2) + (1.0 / (h2 * h2)) * k(setka[0][j][0], setka[0][j][1] + 0.5 * h2);
                pos[j][0] = j;
                A[j][1] = (-2.0 / (h1 * h1)) * k(setka[0][j][0] + 0.5 * h1, setka[0][j][1]);
                pos[j][1] = n + j;
                A[j][2] = (-1.0 / (h2 * h2)) * k(setka[0][j][0], setka[0][j][1] - 0.5 * h2);
                if(j == 0)
                    pos[j][2] = -1;
                else
                    pos[j][2] = j - 1;
                A[j][3] = (-1.0 / (h2 * h2)) * k(setka[0][j][0], setka[0][j][1] + 0.5 * h2);
                if(j == n - 1)
                    pos[j][3] = -3;
                else
                    pos[j][3] = j + 1;
                pos[j][4] = -5;
                B[j] = F(setka[0][j][0], setka[0][j][1]) + (2.0 / h1) * psi(setka[0][j][0], setka[0][j][1], 'l');
            }
        }

        if((nrank <= nproc - 1) && (nrank >= nproc - q))
        {
            #pragma omp for
            for(int j = j_b; j < j_e; j++)
            {
                A[(m-1) * n + j][0] = 1.0 + (2.0 / (h1 * h1)) * k(setka[m-1][j][0] - 0.5 * h1, setka[m-1][j][1]) + (1.0 / (h2 * h2)) * k(setka[m-1][j][0], setka[m-1][j][1] - 0.5 * h2) + (1.0 / (h2 * h2)) * k(setka[m-1][j][0], setka[m-1][j][1] + 0.5 * h2);
                pos[(m-1) * n + j][0] = (m-1) * n + j;
                A[(m-1) * n + j][1] = (-2.0 / (h1 * h1)) * k(setka[m-1][j][0] - 0.5 * h1, setka[m-1][j][1]);
                pos[(m-1) * n + j][1] = (m - 2) * n + j;
                A[(m-1) * n + j][2] = (-1.0 / (h2 * h2)) * k(setka[m-1][j][0], setka[m-1][j][1] - 0.5 * h2);
                if(j == 0)
                    pos[(m-1) * n + j][2] = -1;
                else
                    pos[(m-1) * n + j][2] = (m - 1) * n + j - 1;
                A[(m-1) * n + j][3] = (-1.0 / (h2 * h2)) * k(setka[m-1][j][0], setka[m-1][j][1] + 0.5 * h2);
                if(j == n - 1)
                    pos[(m-1) * n + j][3] = -3;
                else
                    pos[(m-1) * n + j][3] = (m - 1) * n + j + 1;
                pos[(m-1) * n + j][4] = -5;
                B[(m-1) * n + j] = F(setka[m-1][j][0], setka[m-1][j][1]) + (2.0 / h1) * psi(setka[m-1][j][0], setka[m-1][j][1], 'r');
            }
        }

        if(nrank % q == 0)
        {
            #pragma omp for
            for(int i = i_b; i < i_e; i++)
            {
                A[i * n][0] = 1.0 + (2.0 / (h2 * h2)) * k(setka[i][0][0], setka[i][0][1] + 0.5 * h2) + (1.0 / (h1 * h1)) * k(setka[i][0][0] - 0.5 * h1, setka[i][0][1]) + (1.0 / (h1 * h1)) * k(setka[i][0][0] + 0.5 * h1, setka[i][0][1]);
                pos[i * n][0] = i * n;
                A[i * n][1] = (-2.0 / (h2 * h2)) * k(setka[i][0][0], setka[i][0][1] + 0.5 * h2);
                pos[i * n][1] = i * n + 1;
                A[i * n][2] = (-1.0 / (h1 * h1)) * k(setka[i][0][0] - 0.5 * h1, setka[i][0][1]);
                if(i == 0)
                    pos[i * n][2] = -2;
                else
                    pos[i * n][2] = (i - 1) * n;
                A[i * n][3] = (-1.0 / (h1 * h1)) * k(setka[i][0][0] + 0.5 * h1, setka[i][0][1]);
                if(i == m - 1)
                    pos[i * n][3] = -4;
                else
                    pos[i * n][3] = (i + 1) * n;
                pos[i * n][4] = -5;
                B[i * n] = F(setka[i][0][0], setka[i][0][1]) + (2.0 / h2) * psi(setka[i][0][0], setka[i][0][1], 'd');
            }
        }

        if(nrank % q == q - 1)
        {
            #pragma omp for
            for(int i = i_b; i < i_e; i++)
            {
                A[(i + 1) * n - 1][0] = 1.0 + (2.0 / (h2 * h2)) * k(setka[i][n-1][0], setka[i][n-1][1] - 0.5 * h2) + (1.0 / (h1 * h1)) * k(setka[i][n-1][0] - 0.5 * h1, setka[i][n-1][1]) + (1.0 / (h1 * h1)) * k(setka[i][n-1][0] + 0.5 * h1, setka[i][n-1][1]);
                pos[(i + 1) * n - 1][0] = (i + 1) * n - 1;
                A[(i + 1) * n - 1][1] = (-2.0 / (h2 * h2)) * k(setka[i][n-1][0], setka[i][n-1][1] - 0.5 * h2);
                pos[(i + 1) * n - 1][1] = (i + 1) * n - 2;
                A[(i + 1) * n - 1][2] = (-1.0 / (h1 * h1)) * k(setka[i][n-1][0] - 0.5 * h1, setka[i][n-1][1]);
                if(i == 0)
                    pos[(i + 1) * n - 1][2] = -2;
                else
                    pos[(i + 1) * n - 1][2] = i * n - 1;
                A[(i + 1) * n - 1][3] = (-1.0 / (h1 * h1)) * k(setka[i][n-1][0] + 0.5 * h1, setka[i][n-1][1]);
                if(i == m - 1)
                    pos[(i + 1) * n - 1][3] = -4;
                else
                    pos[(i + 1) * n - 1][3] = (i + 2) * n - 1;
                pos[(i + 1) * n - 1][4] = -5;
                B[(i + 1) * n - 1] = F(setka[i][n-1][0], setka[i][n-1][1]) + (2.0 / h2) * psi(setka[i][n-1][0], setka[i][n-1][1], 'u');
            }
        }

        #pragma omp for
        for(int i = i_b; i < i_e; i++)
        {
            #pragma omp collapse(2)
            for(int j = j_b; j < j_e; j++)
            {
                A[i * n + j][0] = 1.0 + (1.0 / (h1 * h1)) * k(setka[i][j][0] - 0.5 * h1, setka[i][j][1]) + (1.0 / (h1 * h1)) * k(setka[i][j][0] + 0.5 * h1, setka[i][j][1]) + (1.0 / (h2 * h2)) * k(setka[i][j][0], setka[i][j][1] - 0.5 * h2) + (1.0 / (h2 * h2)) * k(setka[i][j][0], setka[i][j][1] + 0.5 * h2);
                pos[i * n + j][0] = i * n + j;
                A[i * n + j][1] = (-1.0 / (h1 * h1)) * k(setka[i][j][0] - 0.5 * h1, setka[i][j][1]);
                if(i == 0)
                    pos[i * n + j][1] = -2;
                else
                    pos[i * n + j][1] = (i - 1) * n + j;
                A[i * n + j][2] = (-1.0 / (h1 * h1)) * k(setka[i][j][0] + 0.5 * h1, setka[i][j][1]);
                if(i == m - 1)
                    pos[i * n + j][2] = -4;
                else
                    pos[i * n + j][2] = (i + 1) * n + j;
                A[i * n + j][3] = (-1.0 / (h2 * h2)) * k(setka[i][j][0], setka[i][j][1] - 0.5 * h2);
                if(j == 0)
                    pos[i * n + j][3] = -1;
                else
                    pos[i * n + j][3] = i * n + j - 1;
                A[i * n + j][4] = (-1.0 / (h2 * h2)) * k(setka[i][j][0], setka[i][j][1] + 0.5 * h2);
                if(j == n - 1)
                    pos[i * n + j][4] = -3;
                else
                    pos[i * n + j][4] = i * n + j + 1;
                B[i * n + j] = F(setka[i][j][0], setka[i][j][1]);
            }
        }
    }

    do
    {
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < m * n; i++)
        {
            w_last[i] = w[i];
        }

        if(nrank == 0)
        {
            if((nproc != 1) && (nproc != 2))
            {
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < m; i++)
                {
                    w_up[i] = w[i * n + n - 1];
                }
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < n; i++)
                {
                    w_right[i] = w[(m - 1) * n + i];
                }

                MPI_Send(w_up, m, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
                MPI_Recv(w_up_0, m, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &status);
                MPI_Send(w_right, n, MPI_DOUBLE, q, 2, MPI_COMM_WORLD);
                MPI_Recv(w_right_0, n, MPI_DOUBLE, q, 2, MPI_COMM_WORLD, &status);
            }

            if(nproc == 2)
            {
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < m; i++)
                {
                    w_up[i] = w[i * n + n - 1];
                }

                MPI_Send(w_up, m, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
                MPI_Recv(w_up_0, m, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &status);
            }
        }
        else if(nrank == q - 1)
        {
            if(nproc != 2)
            {
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < m; i++)
                {
                    w_down[i] = w[i * n];
                }
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < n; i++)
                {
                    w_right[i] = w[(m - 1) * n + i];
                }

                MPI_Send(w_down, m, MPI_DOUBLE, q - 2, 2, MPI_COMM_WORLD);
                MPI_Recv(w_down_0, m, MPI_DOUBLE, q - 2, 2, MPI_COMM_WORLD, &status);
                MPI_Send(w_right, n, MPI_DOUBLE, 2 * q - 1, 2, MPI_COMM_WORLD);
                MPI_Recv(w_right_0, n, MPI_DOUBLE, 2 * q - 1, 2, MPI_COMM_WORLD, &status);
            }
            else
            {
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < m; i++)
                {
                    w_down[i] = w[i * n];
                }

                MPI_Send(w_down, m, MPI_DOUBLE, q - 2, 2, MPI_COMM_WORLD);
                MPI_Recv(w_down_0, m, MPI_DOUBLE, q - 2, 2, MPI_COMM_WORLD, &status);
            }
        }
        else if(nrank == nproc - q)
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_up[i] = w[i * n + n - 1];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = w[i];
            }

            MPI_Send(w_up, m, MPI_DOUBLE, nproc - q + 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_up_0, m, MPI_DOUBLE, nproc - q + 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nproc - 2 * q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nproc - 2 * q, 2, MPI_COMM_WORLD, &status);
        }
        else if(nrank == nproc - 1)
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_down[i] = w[i * n];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = w[i];
            }

            MPI_Send(w_down, m, MPI_DOUBLE, nproc - 2, 2, MPI_COMM_WORLD);
            MPI_Recv(w_down_0, m, MPI_DOUBLE, nproc - 2, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nproc - q - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nproc - q - 1, 2, MPI_COMM_WORLD, &status);
        }
        else if((nrank >= 1) && (nrank <= q - 2))
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_up[i] = w[i * n + n - 1];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_right[i] = w[(m - 1) * n + i];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_down[i] = w[i * n];
            }

            MPI_Send(w_up, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_up_0, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_right, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_right_0, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_down, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_down_0, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD, &status);
        }
        else if((nrank <= nproc - 2) && (nrank >= nproc - q + 1))
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_up[i] = w[i * n + n - 1];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_down[i] = w[i * n];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = w[i];
            }

            MPI_Send(w_up, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_up_0, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_down, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_down_0, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD, &status);
        }
        else if(nrank % q == 0)
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_up[i] = w[i * n + n - 1];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_right[i] = w[(m - 1) * n + i];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = w[i];
            }

            MPI_Send(w_up, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_up_0, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_right, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_right_0, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD, &status);
        }
        else if(nrank % q == q - 1)
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_right[i] = w[(m - 1) * n + i];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_down[i] = w[i * n];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = w[i];
            }

            MPI_Send(w_right, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_right_0, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_down, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_down_0, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD, &status);
        }
        else
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_up[i] = w[i * n + n - 1];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_right[i] = w[(m - 1) * n + i];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_down[i] = w[i * n];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = w[i];
            }

            MPI_Send(w_up, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_up_0, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_right, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_right_0, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_down, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_down_0, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD, &status);
        }

        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < m; i++)
        {
            #pragma omp collapse(2)
            for(int j = 0; j < n; j++)
            {
                r[i * n + j] = 0.0;
                #pragma omp collapse(3)
                for (int o = 0; o < 5; o++)
                {
                    if(pos[i * n + j][o] == -5)
                        continue;
                    else if(pos[i * n + j][o] == -1)
                    {
                        r[i * n + j] += A[i * n + j][o] * w_down_0[i];
                    }
                    else if(pos[i * n + j][o] == -2)
                    {
                        r[i * n + j] += A[i * n + j][o] * w_left_0[j];
                    }
                    else if(pos[i * n + j][o] == -3)
                    {
                        r[i * n + j] += A[i * n + j][o] * w_up_0[i];
                    }
                    else if(pos[i * n + j][o] == -4)
                    {
                        r[i * n + j] += A[i * n + j][o] * w_right_0[j];
                    }
                    else
                        r[i * n + j] += A[i * n + j][o] * w_last[pos[i * n + j][o]];
                }

                r[i * n + j] -= B[i * n + j];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(nrank == 0)
        {
            if((nproc != 1) && (nproc != 2))
            {
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < m; i++)
                {
                    w_up[i] = w[i * n + n - 1];
                }
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < n; i++)
                {
                    w_right[i] = w[(m - 1) * n + i];
                }

                MPI_Send(w_up, m, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
                MPI_Recv(w_up_0, m, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &status);
                MPI_Send(w_right, n, MPI_DOUBLE, q, 2, MPI_COMM_WORLD);
                MPI_Recv(w_right_0, n, MPI_DOUBLE, q, 2, MPI_COMM_WORLD, &status);
            }

            if(nproc == 2)
            {
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < m; i++)
                {
                    w_up[i] = w[i * n + n - 1];
                }

                MPI_Send(w_up, m, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
                MPI_Recv(w_up_0, m, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &status);
            }
        }
        else if(nrank == q - 1)
        {
            if(nproc != 2)
            {
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < m; i++)
                {
                    w_down[i] = w[i * n];
                }
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < n; i++)
                {
                    w_right[i] = w[(m - 1) * n + i];
                }

                MPI_Send(w_down, m, MPI_DOUBLE, q - 2, 2, MPI_COMM_WORLD);
                MPI_Recv(w_down_0, m, MPI_DOUBLE, q - 2, 2, MPI_COMM_WORLD, &status);
                MPI_Send(w_right, n, MPI_DOUBLE, 2 * q - 1, 2, MPI_COMM_WORLD);
                MPI_Recv(w_right_0, n, MPI_DOUBLE, 2 * q - 1, 2, MPI_COMM_WORLD, &status);
            }
            else
            {
                #pragma omp parallel for num_threads(4)
                for (int i = 0; i < m; i++)
                {
                    w_down[i] = w[i * n];
                }

                MPI_Send(w_down, m, MPI_DOUBLE, q - 2, 2, MPI_COMM_WORLD);
                MPI_Recv(w_down_0, m, MPI_DOUBLE, q - 2, 2, MPI_COMM_WORLD, &status);
            }
        }
        else if(nrank == nproc - q)
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_up[i] = r[i * n + n - 1];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = r[i];
            }

            MPI_Send(w_up, m, MPI_DOUBLE, nproc - q + 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_up_0, m, MPI_DOUBLE, nproc - q + 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nproc - 2 * q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nproc - 2 * q, 2, MPI_COMM_WORLD, &status);
        }
        else if(nrank == nproc - 1)
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_down[i] = r[i * n];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = r[i];
            }

            MPI_Send(w_down, m, MPI_DOUBLE, nproc - 2, 2, MPI_COMM_WORLD);
            MPI_Recv(w_down_0, m, MPI_DOUBLE, nproc - 2, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nproc - q - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nproc - q - 1, 2, MPI_COMM_WORLD, &status);
        }
        else if((nrank >= 1) && (nrank <= q - 2))
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_up[i] = r[i * n + n - 1];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_right[i] = r[(m - 1) * n + i];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_down[i] = r[i * n];
            }

            MPI_Send(w_up, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_up_0, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_right, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_right_0, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_down, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_down_0, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD, &status);
        }
        else if((nrank <= nproc - 2) && (nrank >= nproc - q + 1))
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_up[i] = r[i * n + n - 1];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_down[i] = r[i * n];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = r[i];
            }

            MPI_Send(w_up, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_up_0, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_down, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_down_0, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD, &status);
        }
        else if(nrank % q == 0)
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_up[i] = r[i * n + n - 1];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_right[i] = r[(m - 1) * n + i];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = r[i];
            }

            MPI_Send(w_up, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_up_0, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_right, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_right_0, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD, &status);
        }
        else if(nrank % q == q - 1)
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_right[i] = r[(m - 1) * n + i];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_down[i] = r[i * n];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = r[i];
            }

            MPI_Send(w_right, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_right_0, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_down, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_down_0, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD, &status);
        }
        else
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_up[i] = r[i * n + n - 1];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_right[i] = r[(m - 1) * n + i];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < m; i++)
            {
                w_down[i] = r[i * n];
            }
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < n; i++)
            {
                w_left[i] = r[i];
            }

            MPI_Send(w_up, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_up_0, m, MPI_DOUBLE, nrank + 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_right, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_right_0, n, MPI_DOUBLE, nrank + q, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_down, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(w_down_0, m, MPI_DOUBLE, nrank - 1, 2, MPI_COMM_WORLD, &status);
            MPI_Send(w_left, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD);
            MPI_Recv(w_left_0, n, MPI_DOUBLE, nrank - q, 2, MPI_COMM_WORLD, &status);
        }

        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < m; i++)
        {
            #pragma omp collapse(2)
            for(int j = 0; j < n; j++)
            {
                Ar[i * n + j] = 0.0;

                #pragma omp collapse(3)
                for (int o = 0; o < 5; o++)
                {
                    if(pos[i * n + j][o] == -5)
                        continue;
                    else if(pos[i * n + j][o] == -1)
                    {
                        Ar[i * n + j] += A[i * n + j][o] * w_down_0[i];
                    }
                    else if(pos[i * n + j][o] == -2)
                    {
                        Ar[i * n + j] += A[i * n + j][o] * w_left_0[j];
                    }
                    else if(pos[i * n + j][o] == -3)
                    {
                        Ar[i * n + j] += A[i * n + j][o] * w_up_0[i];
                    }
                    else if(pos[i * n + j][o] == -4)
                    {
                        Ar[i * n + j] += A[i * n + j][o] * w_right_0[j];
                    }
                    else
                        Ar[i * n + j] += A[i * n + j][o] * r[pos[i * n + j][o]];
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Gatherv(r, m*n, MPI_DOUBLE, r_0, revcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Gatherv(Ar, m*n, MPI_DOUBLE, Ar_0, revcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(nrank == 0)
        {
            tau = skalar(Ar_0, r_0, M + 1, N + 1, h1, h2) / skalar(Ar_0, Ar_0, M + 1, N + 1, h1, h2);
        }

        MPI_Bcast(&tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < m * n; i++)
        {
            w[i] = w_last[i] - tau * r[i];
        }

        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < m * n; i++)
        {
            r[i] = w[i] - w_last[i];
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Gatherv(r, m*n, MPI_DOUBLE, r_0, revcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(nrank == 0)
        {
            cur_eps = sqrt(skalar(r_0, r_0, M + 1, N + 1, h1, h2));
            cout << cur_eps << endl;
        }

        MPI_Bcast(&cur_eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(cur_eps < eps)
        {
            break;
        }
    }
    while(true);

    t = MPI_Wtime() - t1;

    MPI_Reduce(&t, &Time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double Error;

    for(int i = 0; i <= m - 1; i++)
    {
        for(int j = 0; j <= n - 1; j++)
        {
            r[i * n + j] = 2.0 / (1.0 + setka[i][j][0] * setka[i][j][0] + setka[i][j][1] * setka[i][j][1]);
        }
    }

    double error = 0.0;

    for(int i = 0; i < m * n; i++)
    {
        w_last[i] = fabs(w[i] - r[i]);
        if(w_last[i] > error)
        {
            error = w_last[i];
        }
    }

    MPI_Gatherv(w, m*n, MPI_DOUBLE, r_0, revcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(nrank == 0)
    {
        ofstream fout("result.txt");

        for (int i = 0; i < (M + 1) * (N + 1); i++)
        {
            fout << r_0[i] << endl;
        }

        fout.close();

        ofstream fout1("sizes.txt");

        for (int i = 0; i < 2 * nproc; i = i + 2)
        {
            fout1 << Sizes[i] << " " << Sizes[i + 1] << endl;
        }

        fout1.close();
    }

    MPI_Reduce(&error, &Error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if(nrank == 0)
    {
        cout << "Error:   " << Error << endl;
        cout << "Time:    " << Time << endl;
    }

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            delete [] setka[i][j];
        }
    }
    for(int i = 0; i < m; i++)
    {
        delete [] setka[i];
    }
    delete [] setka;
    for(int i = 0; i < m * n; i++)
    {
        delete [] pos[i];
    }
    delete [] pos;
    for(int i = 0; i < m * n; i++)
    {
        delete [] A[i];
    }
    delete [] A;
    delete [] B;
    delete [] r;
    delete [] w;
    delete [] w_last;
    delete [] Ar;
    delete [] Ar_0;
    delete [] r_0;
    delete [] displs;
    delete [] revcounts;
    delete [] w_right;
    delete [] w_left;
    delete [] w_up;
    delete [] w_down;
    delete [] w_right_0;
    delete [] w_left_0;
    delete [] w_up_0;
    delete [] w_down_0;
    delete [] sizes;
    delete [] Sizes;
    delete [] cum_sizes;

    MPI_Finalize();

    return 0;
}
