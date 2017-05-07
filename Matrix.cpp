#include <iostream>
#include "Matrix.h"

Matrix::Matrix(void) { }

Matrix::~Matrix(void) { }

int Matrix::allocateSize(int rows,int cols)
{
	this->rows = rows;
	this->cols = cols;

	float** matrix = new float*[rows];
	if (rows)
	{
	    for (int i = 0; i < rows; i++)
	    {	
	        matrix[i] = new float[cols];
	    }    
	}	
	this->matrix = matrix;
	return 1;
}


int Matrix::fillMatrix(float data[])
{
	this->rows = rows;
	this->cols = cols;

	int k = 0;
	for (int i = 0; i < rows; i++)
    {	
        for (int j = 0; j < cols; j++)
	    {	
	        matrix[i][j] = data[k];
	        k++;
	    }
    }  


	return 1;
}


int Matrix::printMatrix()
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout<<matrix[i][j]<<" ";
        }	
        std::cout<<"\n";
    }	
    std::cout<<"\n\n\n";
    return 1;
}

int Matrix::getrows()
{
	return this->rows;
}

int Matrix::getcols()
{
	return this->cols;
}

float Matrix::get(int x,int y)
{
	return this->matrix[x][y];
}


int Matrix::set(int x,int y,float val)
{
	this->matrix[x][y] = val;
	return 1;
}


Matrix Matrix::transpose()
{
	int size = this->rows * this->cols;
	
	float* tmpdata;
	tmpdata = new float[size];

	int k = 0;
	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			tmpdata[k] = this->matrix[j][i]; k++;
		}	
	}	

	Matrix m2;
	m2.allocateSize(this->cols,this->rows);
	m2.fillMatrix(tmpdata);
	return m2;
}

Matrix Matrix::add(Matrix m)
{
	int size = this->rows * this->cols;
	float* tmpdata;
	tmpdata = new float[size];

	if ( (this->rows == m.rows) && (this->cols == m.cols))
	{
		int k = 0;
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				tmpdata[k] = this->matrix[i][j]+m.matrix[i][j]; k++;
			}	
		}	
	}

	Matrix m2;
	m2.allocateSize(this->rows,this->cols);
	m2.fillMatrix(tmpdata);
	return m2;
}

Matrix Matrix::substract(Matrix m)
{
	int size = this->rows * this->cols;
	float* tmpdata;
	tmpdata = new float[size];

	if ( (this->rows == m.rows) && (this->cols == m.cols))
	{
		int k = 0;
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				tmpdata[k] = this->matrix[i][j]-m.matrix[i][j]; k++;
			}	
		}	
	}

	Matrix m2;
	m2.allocateSize(this->rows,this->cols);
	m2.fillMatrix(tmpdata);
	return m2;
}

Matrix Matrix::substract(int val)
{
	int size = this->rows * this->cols;
	float* tmpdata;
	tmpdata = new float[size];

        int k = 0;    
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                tmpdata[k] = this->matrix[i][j]-val; k++;
            }	
        }

	Matrix m2;
	m2.allocateSize(this->rows,this->cols);
	m2.fillMatrix(tmpdata);
	return m2;
}

Matrix Matrix::substractFrom(int val)
{
	int size = this->rows * this->cols;
	float* tmpdata;
	tmpdata = new float[size];

        int k = 0;    
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                tmpdata[k] = val-this->matrix[i][j]; k++;
            }	
        }

	Matrix m2;
	m2.allocateSize(this->rows,this->cols);
	m2.fillMatrix(tmpdata);
	return m2;
}

Matrix Matrix::scalarMul(float val)
{
	int size = this->rows * this->cols;
	
	float* tmpdata;
	tmpdata = new float[size];

	int k = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			tmpdata[k] = this->matrix[i][j]*val; k++;
		}	
	}	

	Matrix m2;
	m2.allocateSize(this->rows,this->cols);
	m2.fillMatrix(tmpdata);
	return m2;
}

Matrix Matrix::matrixMul(Matrix m)
{
    int size = this->rows * m.cols;
    float* tmpdata;
    tmpdata = new float[size];

    int arrIndx = 0;
    if ( (this->cols == m.rows))
    {
        int p = 0; 
        for(int n = 0; n < this->rows; n++)
        {	
            for(int l = 0; l < m.cols; l++)
            {	
                float tmpSum = 0;
                int j = 0, k = 0;
                for (int i = 0; i < this->cols; i++)
                {
                    tmpSum += this->matrix[k+p][j] * m.matrix[j][k+l];
                    //std::cout<<"( "<< this->matrix[k+p][j] << " * " << m.matrix[j][k+l] << " ) \n";
                    j++;
                }	
                //std::cout<<tmpSum<<"***";	
                tmpdata[arrIndx] = tmpSum;
                arrIndx++;
            }	
            p++;
            //std::cout<<"\n";
        }
        //std::cout<<"\n";	
    }
    Matrix m2;
    m2.allocateSize(this->rows,m.cols);
    m2.fillMatrix(tmpdata);
    return m2;
}

Matrix Matrix::hadamardMul(Matrix m)
{
	int size = this->rows * this->cols;
	float* tmpdata;
	tmpdata = new float[size];

	if ( (this->rows == m.rows) && (this->cols == m.cols))
	{
		int k = 0;
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				tmpdata[k] = this->matrix[i][j]*m.matrix[i][j]; k++;
			}	
		}	
	}

	Matrix m2;
	m2.allocateSize(this->rows,this->cols);
	m2.fillMatrix(tmpdata);
	return m2;
}

Matrix Matrix::kroneckerMul(Matrix m)
{	
	Matrix m2;
	m2.allocateSize((this->rows*m.rows),(this->cols*m.cols));

	Matrix tmpMatrix;
	tmpMatrix.allocateSize(m.rows,m.cols);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			tmpMatrix = m.scalarMul(this->matrix[i][j]);
			//tmpMatrix.printMatrix();
			//std::cout<<"\n";
			for(int k = 0; k < m.rows; k++)
			{
				for(int l = 0; l < m.cols; l++)
				{
					m2.set((i*m.rows)+k,(j*m.cols)+l,tmpMatrix.get(k,l));
				}	
			}

		}	
	}	
	return m2;
}

Matrix Matrix::horizontalConc(Matrix m)
{
	Matrix m2;
	m2.allocateSize((this->rows),(this->cols+m.cols));

	if ( this->rows == m.rows)
	{	
		int k = 0, l = 0, n = 0;
		while (k < 2)
		{	
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					m2.set(i,j+n,this->matrix[i][j]);
				}	
			}

			l = this->rows; n = this->cols; k++;
		}
	}	
	return m2;
}