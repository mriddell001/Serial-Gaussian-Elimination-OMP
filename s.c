/*
Matthew Riddell
mriddell
005498481
CSCI 551
Assignment 4b
Description - This implements a serial version of Gaussian elimination with
              partial pivoting. Now with OMP.
*/
#define _XOPEN_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <time.h>
#include <omp.h>

/**
 * rand_gen - This function generates random numbers when n > 5
 * @param {DOUBLE} a - Matrix to be generated.
 * @param {DOUBLE} b - Backup matrix.
 * @param {DOUBLE} c - Vector of answers to initialize.
 * @param {DOUBLE} d - Vector of l2 to initialize.
 * @param {INT} n - Input size of matrix.
 */
void rand_gen(double *a, double *b, double *c, double *d, int n) {
  srand48(time(0));
  int m = n+1;
  int i = 0;
  #pragma omp parallel for simd schedule(auto)
    for (i = 0; i < m*n; i++) {
      double tmp = drand48();
      double e_o = drand48();
      if (e_o <= 0.5) {tmp = tmp * -1.0;}
      tmp = tmp * 1000000.0;
      *(a + i) = tmp;
      *(b + i) = tmp;
      if (i < n) {
        *(c + i) = 0.0;
        *(d + i) = 0.0;
      }
    }

}

/**
 * calc_ans - This function calculates answer for variables.
 * @param {DOUBLE} a - Matrix to use to calculate answers.
 * @param {DOUBLE} b - Vector of answers.
 * @param {INT} i - Row to be reduced.
 * @param {INT} n - Input size of matrix.
 */
void calc_ans(double *a, double *b, int i, int n) {
  int m = n + 1;
  int j = 0;
  double right_side = *(a + i*m + n);
  double divisor = *(a + i*m + i);
  for (j = n-1; j > i; j--) {
    double element = *(a + i*m + j);
    double variable = *(b + j);
    right_side = right_side - element * variable;
  }
  *(b + i) = right_side / divisor;
}

/**
 * elim_rows - This function elims rowly.
 * @param {DOUBLE} a - Matrix to be generated.
 * @param {INT} i - Row to be eliminated.
 * @param {INT} divisor - Value to divise by.
 * @param {INT} n - Input size of matrix.
 */
void elim_rows(double *a, double *b, int i, double divisor, int n) {
  int m = n + 1;
  int j = 0;
  int k = 0;
  int l = 0;

  #pragma omp parallel for schedule(auto)
    for (l = 0; l < n; l++) {
      *(b + l) = *(a + l*m + i);
    }

  #pragma omp parallel for schedule(auto)
    for (j = i+1; j < n; j++) {
      for (k = i; k < m; k++) {
        double s_elim = *(a + i*m + k);
        double c_elim = *(a + j*m + k);
        double scaler = *(b + j)/divisor;
        s_elim = s_elim * scaler;
        *(a + j*m + k) = c_elim - s_elim;
      }
    }
}

/**
 * find_max - This function finds the largest value in the associated column.
 * @param {DOUBLE} a - Matrix to be searched.
 * @param {INT} i - Column to be searched.
 * @param {INT} n - Input size of matrix.
 * @returns {INT} - returns the row containing the largest value in column i.
 */
int find_max(double *a, int i, int n) {
  int m = n+1;
  int max = i;
  int j = 0;
  double max_value = *(a + i*m + i);
  max_value = fabs(max_value);
  for (j = i+1; j < n; j++) {
    double tmp = *(a + j*m + i);
    tmp = fabs(tmp);
    if (tmp > max_value) {
      max_value = tmp;
      max = j;
    }
  }
  return max;
}

/**
 * sum_of_squares - This function calculates the sum of squares.
 * @param {DOUBLE} ortrix - Backup matrix
 * @param {DOUBLE} answers - Vector of answers
 * @param {DOUBLE} l2trix - Vector of l2
 * @param {INT} n - Input size of matrix.
 */
void sum_of_squares(double *ortrix, double *answers, double *l2trix, int n) {
  int m = n + 1;
  int i = 0;
  int j = 0;
  double sum_sqrs = 0.0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      double a = *(ortrix + i*m + j);
      double x = *(answers + j);
      double y = a*x;
      *(l2trix + i) = *(l2trix + i) + y;
    }
    *(l2trix + i) = *(l2trix + i) - *(ortrix + i*m + n);
    sum_sqrs = sum_sqrs + pow(*(l2trix + i), 2.0);
  }
  printf("%.10e\n", sqrt(sum_sqrs));
}

/**
 * swap_rows - This function swaps rows in the matrix.
 * @param {DOUBLE} a - Matrix to be row swapped.
 * @param {INT} i - Row value.
 * @param {INT} j - Column value.
 * @param {INT} n - Input size of matrix.
 */
void swap_rows(double *a, int i, int j, int n) {
  int m = n + 1;
  int k = 0;
  #pragma omp parallel for simd schedule(auto)
    for (k = 0; k < m; k++) {
      double tmp = *(a + i*m + k);
      *(a + i*m + k) = *(a + j*m + k);
      *(a + j*m + k) = tmp;
    }
}

/**
 * print_answers - This function prints the original matrix.
 * @param {DOUBLE} answers - Answer matrix
 * @param {INT} n - Input size of matrix.
 */
void print_answers(double *answers, int n) {
  int i = 0;
  for (i = 0; i < n; i++) {
    printf("%.10e", *(answers + i));
    if (i < n+1) {
      printf(" ");
    }
  }
  printf("\n");
}

/**
 * print_ortrix - This function prints the original matrix.
 * @param {DOUBLE} ortrix - Original matrix
 * @param {INT} n - Input size of matrix.
 */
void print_ortrix(double *ortrix, int n) {
  int i = 0;
  int j = 0;
  int m = n + 1;
  printf("\n");
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      printf("%.10e", *(ortrix + i*m + j));
      if (j<m-1) {printf(" ");}
    }
    printf("\n");
  }
  printf("\n");
}

/**
 * main - This function does the thing.
 * @param {INT} argc - Number of arguments.
 * @param {INT} argv - Array of arguments.
 * @returns {INT} - returns zero.
 */
int main(int argc, char const *argv[]) {
  double start_time = omp_get_wtime();
  struct timeval user_time, sys_time;
  struct rusage ru;
  int n = atoi(argv[1]);
  int m = n + 1;
  int procs, threads;

  double *matrix = NULL, *ortrix = NULL, *l2trix = NULL, *answers = NULL, *mult = NULL;
  while (matrix == NULL) {matrix = (double*)_mm_malloc(m * n * sizeof (double),64);}
  while (ortrix == NULL) {ortrix = (double*)_mm_malloc(m * n * sizeof (double),64);}
  while (l2trix == NULL) {l2trix = (double*)_mm_malloc(n * sizeof (double),64);}
  while (answers == NULL) {answers = (double*)_mm_malloc(n * sizeof (double),64);}
  while (mult == NULL) {mult = (double*)_mm_malloc(n * sizeof (double),64);}
  int i = 0;
  int j = 0;
  if (n < 5) {
    for (i = 0; i < n; i++) {
      for (j = 0; j < m; j++) {
        double tmp = 0.0;
        scanf("%lf", &tmp);
        *(matrix + i*m + j) = tmp;
        *(ortrix + i*m + j) = tmp;
      }
      *(l2trix + i) = 0.0;
      *(answers + i) = 0.0;
    }
  }
  else {rand_gen(matrix, ortrix, l2trix, answers, n);}
  #pragma omp parallel
        #pragma omp master
        {
          threads = omp_get_num_threads();
          procs = omp_get_num_procs();
        }
  /*
   * The following section performs the gaussian elimination on each row. It
   * selects the absolute largest value in the column and swaps (if needed) the
   * current row with the row with the absolute greatest value. Next it takes
   * the divisor value and if that value is non-zero then it uses it to perform
   * the elimination of the columns below. If i = n-2 then a final set of
   * calculations are performed to result in a correct gaussian elimination.
   */
  for (i = 0; i < n-1; i++) {
    int max = find_max(matrix, i, n);
    if (max > i) {swap_rows(matrix, i, max, n);}
    double divisor = *(matrix + i*m + i);
    if (divisor != 0.0) { //As long as at least one row had a non-zero value
      elim_rows(matrix, mult, i, divisor, n);
    }
  }
  for (i = n-1; i > -1; i--) {
    calc_ans(matrix, answers, i, n);
  }
  double end_time = omp_get_wtime();
  double total_time = end_time - start_time;
  printf("%.10e %d %d ", total_time, procs, threads);
  sum_of_squares(ortrix, answers, l2trix, n);
  if (n < 5) {
    print_ortrix(ortrix, n);
    print_answers(answers, n);
  }
  return 0;
}
