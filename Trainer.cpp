/* 
 * File:   Trainer.cpp
 * Author: heshan
 * 
 * Created on May 6, 2017, 5:37 PM
 */

#include <iostream>

#include "Trainer.h"
#include "Matrix.h"
#include "Activation.h"

Trainer::Trainer() {
    
    int classes = 26;
    int chars = 8;
    
    
    // Initializing the input Matrix
    
    
    
    // Initializing the target Matrix
    float targetMatrixData[classes*chars];
    int k = 0;
    for (int i = 0; i < classes; i++){
        for (int j = 0; j < chars; j++){
            if ( i == j ) targetMatrixData[k] = 1;
            else targetMatrixData[k] = 0;
        }
    }
    targetMatrix.allocateSize(classes,chars);
    targetMatrix.fillMatrix(targetMatrixData);	
    
    
    
    inputMatrix      
}

Trainer::Trainer(const Trainer& orig) { }

Trainer::~Trainer() { }


