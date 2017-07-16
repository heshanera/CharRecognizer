/* 
 * File:   main.cpp
 * Author: heshan
 *
 * Created on May 4, 2017, 11:38 PM
 */

#include <cstdlib>
#include <iostream>

#include "Recognizer.h"
#include "ImageProcessor.h"

/*
 * 
 */
int main(int argc, char** argv) {

    Recognizer rc;
    //rc.train(35);
    
    rc.recognize("imgs/testing/A.png");
    rc.recognize("imgs/testing/A2.png");
    rc.recognize("imgs/testing/A3.png");
    rc.recognize("imgs/testing/A4.png");
    rc.recognize("imgs/testing/D.png");
    rc.recognize("imgs/testing/C.png");
    rc.recognize("imgs/testing/D.png");
    rc.recognize("imgs/testing/s.png");
    rc.recognize("imgs/testing/M2.png");
    rc.recognize("imgs/testing/V.png");
    rc.recognize("imgs/testing/H.png");
    rc.recognize("imgs/testing/W.png");
    
    
    return 0;
}

