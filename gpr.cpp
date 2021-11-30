// GridStage1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <sys/time.h>
#include <omp.h>
#include <algorithm>
#include <string>
#include <math.h>


// #define PRINT_MATRIX


void printMatrix(const double matrix[], const uint64_t iSize, const uint64_t jSize)
{
    uint64_t index = 0;
    for (int i = 0; i < iSize; i++)
    {
        for (int j = 0; j < jSize; j++)
        {
            printf("  %lf", matrix[index]);
            index++;
        }
        printf("\n");
    }
}



double get5HundredthsRandom(void)
{
    return  (0.100 * (double)rand() / (double)RAND_MAX) - 0.05 ;
}


struct MatrixDataForMultiplication
{
    // matrixPointer
    // iSize: the number of rows for area of interest
    // jSize: the number of columns  for area of interest
    double* matrix;
    uint64_t iSize, jSize;
};


void multiplyMatrixParallel(const MatrixDataForMultiplication& matrixA,
                           const MatrixDataForMultiplication& matrixB,
                           MatrixDataForMultiplication& result)
{
    result.iSize = matrixA.iSize;
    result.jSize = matrixB.jSize;
    #pragma omp parallel shared(matrixA, matrixB, result)
    {
        #pragma omp for nowait collapse(2)
        for (uint64_t i = 0; i < result.iSize; i++)
        {
            for (uint64_t j = 0; j < result.jSize; j++)
            {
                result.matrix[i * result.jSize + j] = 0;
                for (uint64_t k = 0; k < matrixA.jSize; k++)
                {
                    // printf("%.2lf X %.2lf + ", matrixA[matrixAIndex], matrixB[matrixBIndex]);
                    result.matrix[i * result.jSize + j] += matrixA.matrix[i * matrixA.jSize + k] * matrixB.matrix[k * matrixB.jSize + j];
                }
            }
        }
    }
}



void computeLUMatrixCompact(double matrix[], const uint64_t n)
{
    // for (uint64_t k = 0; k < n - 1; k++)
    // {
    //     #pragma omp parallel shared(matrix, k, n)
    //     {
    //         #pragma omp for nowait schedule(static)
    //             for (uint64_t i = k + 1; i < n; i++)
    //             {
    //                 double toBeL = matrix[i * n + k] / matrix[k * n + k];
    //                 matrix[i * n + k] = toBeL;
    //                 for (uint64_t j = k + 1; j < n; j++)
    //                 {
    //                     matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
    //                 }
    //             }
    //     }
    // }

    for (uint64_t k = 0; k < n - 1; k++)
    {
        uint64_t chunkSize = 100000;
        if( (chunkSize / (n - k)) < 20)
        {
            chunkSize = chunkSize / (n - k);
            #pragma omp parallel shared(matrix, k, n, chunkSize)
            {
                #pragma omp for nowait schedule(static,chunkSize)
                    for (uint64_t i = k + 1; i < n; i++)
                    {
                        double toBeL = matrix[i * n + k] / matrix[k * n + k];
                        matrix[i * n + k] = toBeL;
                        for (uint64_t j = k + 1; j < n; j++)
                        {
                            matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
                        }
                    }
            }
        }
        else
        {
            #pragma omp parallel shared(matrix, k, n)
            {
                #pragma omp for nowait schedule(static)
                    for (uint64_t i = k + 1; i < n; i++)
                    {
                        double toBeL = matrix[i * n + k] / matrix[k * n + k];
                        matrix[i * n + k] = toBeL;
                        for (uint64_t j = k + 1; j < n; j++)
                        {
                            matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
                        }
                    }
            }
        }
    }
}



void extractLUMartixFromCompactResult(double matrix[], double matrixL_[], double matrixU_[], const uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
    {
        for (uint64_t j = i; j < n; j++)
        {
            matrixU_[i * n + j] = matrix[i * n + j];
        }
    }

    for (int64_t j = 0; j < n; j++)
    {
        matrixL_[j * n + j] = 1;
        for (int64_t i = j + 1; i < n; i++)
        {
            matrixL_[i * n + j] = matrix[i * n + j];
        }
    }
}



// AX = B
// A = LU
// LUX = B
// We can solve that by solving the 2 below
// 1) LY = B
// 2) UX = Y
// In this function solving for 1)
// size of compactMatrix is nXn while B and Y are nX1,  Y is unkown
void substitutionForL(double compactMatrix_[], double matrixB_[], double matrixY_[], const uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
    {
        double rightSide = matrixB_[i]; // we make it so that "Y[i] = rightSide" mathematically
        for (uint64_t j = 0; j < i; j++)
        {
            rightSide -= compactMatrix_[i * n + j] * matrixY_[j];
        }
        matrixY_[i] = rightSide;
    }
}


// AX = B
// A = LU
// LUX = B
// We can solve that by solving the 2 below
// 1) LY = B
// 2) UX = Y
// In this function solving for 2)
// size of compactMatrix is nXn while y and X are nX1, X is unkown
void substitutionForU(double compactMatrix_[], double matrixY_[], double matrixX_[], const uint64_t n)
{
    for (int64_t i = n - 1; i > -1; i--)
    {
        double rightSide = matrixY_[i]; // we make it so that "Y[i] = rightSide" mathematically
        for (uint64_t j = n - 1; j > i; j--)
        {
            rightSide -= compactMatrix_[i * n + j] * matrixX_[j];
        }
        matrixX_[i] = rightSide / compactMatrix_[i * n + i];
    }
}



int main(int argc, char* argv[])
{
    double predictionPointX = 0, predictionPointY = 0;
    uint64_t gridM = 0;
    if (argc < 4)
    {
        printf("Not enough arguments were provided!!! Make sure you add grid size and your points x and i");
        return 0;
    }
    else
    {
        gridM = std::stol(argv[1], nullptr, 10);
        predictionPointX = std::stod(argv[2]);
        predictionPointY = std::stod(argv[3]);
    }


    struct timeval start, stop;
    double total_time;


    gettimeofday(&start, NULL); 

    double gridH = 1.0 / (1.0 + gridM);


    double* gridPosition = new double[gridM];
    for (uint64_t i = 0; i < gridM; i++)
    {
        gridPosition[i] = gridH * (i + 1);
    }


    double* gridPositionSubtractedFromHalfAndSquared = new double[gridM];
    for (uint64_t i = 0; i < gridM; i++)
    {
        gridPositionSubtractedFromHalfAndSquared[i] = gridPosition[i] - 0.5;
        gridPositionSubtractedFromHalfAndSquared[i] = gridPositionSubtractedFromHalfAndSquared[i] * gridPositionSubtractedFromHalfAndSquared[i];
    }

    double* observedValues = new double[gridM * gridM];
    #pragma omp parallel shared(observedValues, gridPositionSubtractedFromHalfAndSquared, gridM)
    {
            #pragma omp for nowait 
            for (uint64_t i = 0; i < gridM; i++)
            {
                for (uint64_t j = i; j < gridM; j++)
                {
                    double withoutNoise = 1 - (gridPositionSubtractedFromHalfAndSquared[i] + gridPositionSubtractedFromHalfAndSquared[j]);
                    observedValues[i * gridM + j] = withoutNoise + get5HundredthsRandom();
                    observedValues[j * gridM + i] = withoutNoise + get5HundredthsRandom();
                }
            }
    }

    #ifdef PRINT_MATRIX
    printMatrix(observedValues, gridM * gridM, 1);
    #endif

    // finding K which is n X n
    double* matrixKCaptial = new double[gridM * gridM * gridM * gridM];
     #pragma omp parallel shared(matrixKCaptial, gridPosition, gridM)
    {
        #pragma omp for nowait collapse(2) schedule(static,50)
        for (uint64_t i = 0; i < gridM; i++)
        {
            for (uint64_t j = 0; j < gridM; j++)
            {
                for (uint64_t k = i; k < gridM; k++)
                {
                    for (uint64_t l = 0; l < gridM; l++)
                    {
                        double xDistancePart = gridPosition[i] - gridPosition[k];
                        xDistancePart = xDistancePart * xDistancePart;
                        double yDistancePart = gridPosition[j] - gridPosition[l];
                        yDistancePart = yDistancePart * yDistancePart;
                        double totalDistance = yDistancePart + xDistancePart;
                        double kValue = exp(-totalDistance);
                        matrixKCaptial[i* gridM* gridM* gridM + j* gridM* gridM + k* gridM + l] = kValue;
                        matrixKCaptial[k * gridM * gridM * gridM + l * gridM * gridM + i * gridM + j] = kValue;
                    }
                }
            }
        }
    }



    // adding noise
    for (uint64_t i = 0; i < gridM * gridM; i++)
    {
        matrixKCaptial[i * gridM * gridM + i] += 0.01;
    }

    #ifdef PRINT_MATRIX
    printMatrix(matrixKCaptial, gridM * gridM, gridM * gridM);
    #endif

    // small K transpose
    double* kTranspose = new double[gridM * gridM];
    #pragma omp parallel shared(kTranspose, gridPosition, gridM, predictionPointX, predictionPointY)
    {
        #pragma omp for nowait 
        for (uint64_t i = 0; i < gridM; i++)
        {
            for (uint64_t j = 0; j < gridM; j++)
            {
                double xDistancePart = gridPosition[i] - predictionPointX;
                xDistancePart = xDistancePart * xDistancePart;
                double yDistancePart = gridPosition[j] - predictionPointY;
                yDistancePart = yDistancePart * yDistancePart;
                double totalDistance = yDistancePart + xDistancePart;
                double kValue = exp(-totalDistance);
                kTranspose[i * gridM + j] = kValue;
            }
        }
    }


    #ifdef PRINT_MATRIX
    printMatrix(kTranspose, gridM * gridM, 1);
    #endif



    gettimeofday(&stop, NULL); 
    total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);
    #ifdef PRINT_MATRIX
    printf("Full initiation (%d X %d): time it took %lf\n", gridM, gridM, total_time);
    #endif


    gettimeofday(&start, NULL); 
    computeLUMatrixCompact(matrixKCaptial, gridM * gridM);
    double* matrixY = new double[gridM * gridM]; // gridM * gridM X 1
    substitutionForL(matrixKCaptial, observedValues, matrixY, gridM * gridM);
    double* matrixX = new double[gridM * gridM]; // gridM * gridM X 1 or matrix z if you will
    substitutionForU(matrixKCaptial, matrixY, matrixX, gridM * gridM);


    struct MatrixDataForMultiplication kTransposeCalculatedMultiplication = {kTranspose, 1, gridM * gridM};
    struct MatrixDataForMultiplication matrixXMultiplication = {matrixX, gridM * gridM, 1};
    double* result = new double[1]; // gridM * gridM X 1 or matrix z if you will
    struct MatrixDataForMultiplication finalResultMultiplication = {result, 1, 1};

    multiplyMatrixParallel(kTransposeCalculatedMultiplication, matrixXMultiplication, finalResultMultiplication);
    gettimeofday(&stop, NULL); 
    total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);

    printf("Total time = %lf seconds, Predicted Value = %lf", total_time, result[0]);

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file




// functions used for testing:

// bool compareMatrix5Digit(const double matrix1[], const double matrix2[], const uint64_t iSize, const uint64_t jSize)
// {
//     uint64_t index = 0;
//     for (uint64_t i = 0; i < iSize; i++)
//     {
//         for (uint64_t j = 0; j < jSize; j++)
//         {
//             double delta = matrix1[index] - matrix2[index];
//             if (delta < 0)
//                 delta = -delta;
//             if (delta > 0.000009)
//             {
//                 return false;
//             }
//             index++;
//         }
//     }
//     return true;
// }

// void duplicateMatrix(const double matrixSource[], double matrixDestination[], const uint64_t iSize, const uint64_t jSize)
// {
//     std::copy(&matrixSource[0], &matrixSource[iSize * jSize], &matrixDestination[0]);
// }

// void generateMatrix(double matrix[], const uint64_t iSize, const uint64_t jSize)
// {
//     uint64_t index = 0;
//     for (uint64_t i = 0; i < iSize; i++)
//     {
//         for (uint64_t j = 0; j < jSize; j++)
//         {
//             matrix[index] = (double)rand() / (double)RAND_MAX;
//             index++;
//         }
//     }
// }


// void computeLUMatrixCompactV1(double matrix[], const uint64_t n)
// {
//     for (uint64_t k = 0; k < n - 1; k++)
//     {
//         #pragma omp parallel shared(matrix, k, n)
//         {
//             #pragma omp for nowait 
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                     matrix[i * n + k] = toBeL;
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
//                     }
//                     // matrix[i * n + k] = toBeL;
//                 }
//         }
//     }
// }



// void computeLUMatrixCompactV2(double matrix[], const uint64_t n)
// {
//     for (uint64_t k = 0; k < n - 1; k++)
//     {
//         #pragma omp parallel shared(matrix, k, n)
//         {
//             #pragma omp for nowait schedule(static,10)
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                     matrix[i * n + k] = toBeL;
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
//                     }
//                     // matrix[i * n + k] = toBeL;
//                 }
//         }
//     }
// }



// void computeLUMatrixCompactV3(double matrix[], const uint64_t n)
// {
//     for (uint64_t k = 0; k < n - 1; k++)
//     {
//         #pragma omp parallel shared(matrix, k, n)
//         {
//             #pragma omp for nowait schedule(dynamic)
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     matrix[i * n + k] /= matrix[k * n + k];
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] -= (matrix[i * n + k] * matrix[k * n + j]);
//                     }
//                 }
//         }
//     }
// }


// void computeLUMatrixCompactV4(double matrix[], const uint64_t n)
// {
//     for (uint64_t k = 0; k < n - 1; k++)
//     {
//         uint64_t i;
//         #pragma omp parallel shared(matrix, k, n)
//         {
//             #pragma omp for nowait private(i)
//                 for (i = k + 1; i < n; i++)
//                 {
//                     matrix[i * n + k] /= matrix[k * n + k];
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] -= (matrix[i * n + k] * matrix[k * n + j]);
//                     }
//                 }
//         }
//     }
// }




// void computeLUMatrixCompactV5(double matrix[], const uint64_t n)
// {
//     double *tempArray = new double[n];
//     for (uint64_t k = 0; k < n - 1; k++)
//     {
//         #pragma omp parallel shared(matrix, k, n)
//         {
//             #pragma omp for nowait collapse(2) 
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     // double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                     // matrix[i * n + k] = toBeL;
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] = matrix[i * n + j] - matrix[i * n + k] / matrix[k * n + k] * matrix[k * n + j];
//                     }
//                     // matrix[i * n + k] = toBeL;
//                 }
//         }
        
//         for (uint64_t i = k + 1; i < n; i++)
//         {
//             matrix[i * n + k] = matrix[i * n + k] / matrix[k * n + k];
//         }
//     }
// }



// void computeLUMatrixCompactV6(double matrix[], const uint64_t n)
// {
//     double *tempArray = new double[n];
//     for (uint64_t k = 0; k < n - 1; k++)
//     {
//         #pragma omp parallel shared(matrix, k, n)
//         {
//             #pragma omp for nowait collapse(2) schedule(static,10)
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     // double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                     // matrix[i * n + k] = toBeL;
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] = matrix[i * n + j] - matrix[i * n + k] / matrix[k * n + k] * matrix[k * n + j];
//                     }
//                     // matrix[i * n + k] = toBeL;
//                 }
//         }
//         for (uint64_t i = k + 1; i < n; i++)
//         {
//             matrix[i * n + k] = matrix[i * n + k] / matrix[k * n + k];
//         }
//     }
// }


// void computeLUMatrixCompactV7(double matrix[], const uint64_t n)
// {
//     double *tempArray = new double[n];
//     for (uint64_t k = 0; k < n - 1; k++)
//     {
//         #pragma omp parallel shared(matrix, k, n)
//         {
//             #pragma omp for nowait collapse(2) schedule(dynamic)
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     // double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                     // matrix[i * n + k] = toBeL;
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] = matrix[i * n + j] - matrix[i * n + k] / matrix[k * n + k] * matrix[k * n + j];
//                     }
//                     // matrix[i * n + k] = toBeL;
//                 }
//         }
//         for (uint64_t i = k + 1; i < n; i++)
//         {
//             matrix[i * n + k] = matrix[i * n + k] / matrix[k * n + k];
//         }
//     }
// }



// void computeLUMatrixCompactV8(double matrix[], const uint64_t n)
// {
//     #pragma omp parallel
//     {
//         for (uint64_t k = 0; k < n - 1; k++)
//         {
//             #pragma omp for 
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                     matrix[i * n + k] = toBeL;
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
//                     }
//                     // matrix[i * n + k] = toBeL;
//                 }
//         }
//     }
// }



// void computeLUMatrixCompactV9(double matrix[], const uint64_t n)
// {
//     #pragma omp parallel
//     {
//         for (uint64_t k = 0; k < n - 1; k++)
//         {
//             #pragma omp for nowait
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                     matrix[i * n + k] = toBeL;
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
//                     }
//                     // matrix[i * n + k] = toBeL;
//                 }

//             #pragma omp barrier
//         }
//     }
// }



// void computeLUMatrixCompactV10(double matrix[], const uint64_t n)
// {
//     for (uint64_t k = 0; k < n - 1; k++)
//     {
//         #pragma omp parallel shared(matrix, k, n)
//         {
//             #pragma omp for nowait schedule(static,50)
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                     matrix[i * n + k] = toBeL;
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
//                     }
//                     // matrix[i * n + k] = toBeL;
//                 }
//         }
//     }
// }




// void computeLUMatrixCompactV11(double matrix[], const uint64_t n)
// {
//     for (uint64_t k = 0; k < n - 1; k++)
//     {
//         #pragma omp parallel shared(matrix, k, n)
//         {
//             #pragma omp for nowait schedule(static,25)
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                     matrix[i * n + k] = toBeL;
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
//                     }
//                     // matrix[i * n + k] = toBeL;
//                 }
//         }
//     }
// }




// void computeLUMatrixCompactV12(double matrix[], const uint64_t n)
// {
//     for (uint64_t k = 0; k < n - 1; k++)
//     {
//         #pragma omp parallel shared(matrix, k, n)
//         {
//             #pragma omp for nowait schedule(static,75)
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                     matrix[i * n + k] = toBeL;
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
//                     }
//                     // matrix[i * n + k] = toBeL;
//                 }
//         }
//     }
// }



// void computeLUMatrixCompactV13(double matrix[], const uint64_t n)
// {
//     for (uint64_t k = 0; k < n - 1; k++)
//     {
//         #pragma omp parallel shared(matrix, k, n)
//         {
//             #pragma omp for nowait schedule(static,100)
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                     matrix[i * n + k] = toBeL;
//                     for (uint64_t j = k + 1; j < n; j++)
//                     {
//                         matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
//                     }
//                     // matrix[i * n + k] = toBeL;
//                 }
//         }
//     }
// }


// void computeLUMatrixCompactV14(double matrix[], const uint64_t n)
// {
//     #pragma omp parallel shared (matrix, n)
//     {
//         #pragma omp single
//         { 
//             for (uint64_t k = 0; k < n - 1; k++)
//             {
//                 for (uint64_t i = k + 1; i < n; i++)
//                 {
//                     #pragma omp task shared(matrix) firstprivate(i, k, n)
//                     { 
//                         double toBeL = matrix[i * n + k] / matrix[k * n + k];
//                         matrix[i * n + k] = toBeL;
//                         for (uint64_t j = k + 1; j < n; j++)
//                         {
//                             matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
//                         }
//                     }
//                 }
//                 #pragma omp taskwait
//             }
//         }
//     }
// }
