/* 
 * File:   Matrix.h
 * Author: heshan
 *
 * Created on May 4, 2017, 8:31 PM
 */

#ifndef MATRIX_H
#define MATRIX_H

class Matrix{
public:
    float** matrix;
    int rows, cols;

    int allocateSize(int rows,int cols); 
    int fillMatrix(float []);
    int printMatrix();

    int getrows();
    int getcols();

    float get(int x,int y);
    int set(int x,int y, float val);

    Matrix transpose(); //matrix transposition
    Matrix add(Matrix m); //matrix addition
    Matrix subtract(Matrix m); //matrix subtraction
    Matrix subtract(int val); //matrix subtraction by a scalar (Matrix - 1)
    Matrix subtractFrom(int val); //matrix subtraction by a scalar (1 - Matrix)
    Matrix scalarMul(float val); //matrix multiplication with a scalar
    Matrix scalarDiv(float val); //matrix division with a scalar
    Matrix matrixMul(Matrix m); //ordinary matrix multiplication
    Matrix hadamardMul(Matrix m); //Hadamard multiplication (component-wise multiplication)
    Matrix kroneckerMul(Matrix m); //Kronecker multiplication
    Matrix horizontalConc(Matrix m); //horizontal matrix concatenation


    Matrix();
    ~Matrix(void);
};

#endif /* MATRIX_H */

