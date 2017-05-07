/* 
 * File:   Activation.h
 * Author: heshan
 *
 * Created on May 4, 2017, 8:36 PM
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"

class Activation {
public:
    Activation();
    Activation(const Activation& orig);
    virtual ~Activation();
    
    /****** activation functions *************************/

    // Unit step
    static float unitStep(float x);
    static Matrix unitStep(Matrix m);

    // Linear
    static float linear(float m, float c, float x);
    static Matrix linear(float m, float c, Matrix x);

    // Sigmoid
    static float sigmoid(float x);
    static Matrix sigmoid(Matrix m);

    // Piecewise Linear 

    // Gaussian

    /****** activation functions derivatives *************/

    // Sigmoid
    static float SigmoidDerivative(float x);
    static Matrix SigmoidDerivative(Matrix m);
    
private:

};

#endif /* ACTIVATION_H */

