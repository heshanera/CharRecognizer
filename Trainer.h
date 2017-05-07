/* 
 * File:   Trainer.h
 * Author: heshan
 *
 * Created on May 6, 2017, 5:37 PM
 */

#ifndef TRAINER_H
#define TRAINER_H

#include "Matrix.h"

class Trainer {
public:
    Trainer();
    Trainer(const Trainer& orig);
    virtual ~Trainer();
    
    int initializeWeightMatrices();
    int forwardPropagation();
    int backPropagation();
    
private:
    Matrix inputMatrix, targetMatrix;
    Matrix weightMatrix1, weightMatrix2, weightMatrix3;
    Matrix hiddenLayer1Matrix, hiddenLayer2Matrix;
    Matrix outputLayerMatrix;
    Matrix errorMarginMatrix;

};

#endif /* TRAINER_H */

