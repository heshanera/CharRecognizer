/* 
 * File:   Matrix.h
 * Author: heshan
 *
 * Created on May 4, 2017, 8:31 PM
 */

#ifndef MATRIX_H
#define MATRIX_H

class Matrix {
public:
    Matrix();
    Matrix(const Matrix& orig);
    virtual ~Matrix();
    
    int allocateSize(int rows,int cols); 
    int fillMatrix(float []);
    int printMatrix();
    int getrows();
    int getcols();
    float get(int x,int y);
    int set(int x,int y, float val);

    Matrix transpose(); //matrix transposition 
    Matrix add(Matrix m); //matrix addition
    Matrix substract(Matrix m); //matrix addition
    Matrix scalarMul(float val); //matrix multiplication with a scalar
    Matrix matrixMul(Matrix m); //ordinary matrix multiplication
    Matrix hadamardMul(Matrix m); //Hadamard multiplication (component-wise multiplication)
    Matrix kroneckerMul(Matrix m); //Kronecker multiplication
    Matrix horizontalConc(Matrix m); //horizontal matrix concatenation
    
private:
    float** matrix;
    int rows, cols;
};

#endif /* MATRIX_H */

