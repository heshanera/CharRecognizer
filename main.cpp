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
    rc.train(200);//(36);
    rc.recognize("imgs/testing/A.png");
    rc.recognize("imgs/testing/A2.png");
    rc.recognize("imgs/testing/A3.png");
    rc.recognize("imgs/testing/A4.png");
    
    
    
    rc.recognize("imgs/testing/C2.png");
    rc.recognize("imgs/testing/s.png");
    rc.recognize("imgs/testing/M2.png");
    rc.recognize("imgs/testing/V.png");
    
    rc.recognize("imgs/testing/01.png");
    rc.recognize("imgs/testing/11.png");
    rc.recognize("imgs/testing/21.png");
    rc.recognize("imgs/testing/31.png");
    rc.recognize("imgs/testing/41.png");
    rc.recognize("imgs/testing/51.png");
    rc.recognize("imgs/testing/61.png");
    rc.recognize("imgs/testing/71.png");
    rc.recognize("imgs/testing/81.png");
    rc.recognize("imgs/testing/91.png");
    
    
    // writing the ranges to the file
    
    
    return 0;
}

