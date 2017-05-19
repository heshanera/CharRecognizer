/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

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
    //rc.train(18);//(36);
    //rc.recognize("imgs/testing/A.png");
    rc.recognize("imgs/testing/A.png");
    rc.recognize("imgs/testing/A2.png");
    rc.recognize("imgs/testing/A3.png");
    rc.recognize("imgs/testing/A4.png");
    rc.recognize("imgs/testing/C2.png");
    //rc.recognize("imgs/testing/s.png");
    //rc.recognize("imgs/testing/M2.png");
    //rc.recognize("imgs/testing/V.png");
    
    
    // writing the ranges to the file
    
    
    return 0;
}

