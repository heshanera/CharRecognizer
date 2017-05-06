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
    
private:
    Matrix  inputMatrix,
            weightMatrix1,
            weightMatrix2,
            weightMatrix3,
            targetMatrix;

};

#endif /* TRAINER_H */

