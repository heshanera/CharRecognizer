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

/*
 * 
 */
int main(int argc, char** argv) {

    Recognizer rc;
    rc.recognize("imgs/A.jpg");
    rc.recognize("imgs/B.jpg");
    rc.recognize("imgs/C.jpg");
    rc.recognize("imgs/D.jpg");
    
    return 0;
}

