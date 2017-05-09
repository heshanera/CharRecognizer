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

    //Recognizer rc;
    //rc.train();
    //rc.recognize("imgs/testing/Italic_A.jpg");
    //rc.recognize("imgs/testing/A3.jpg");
    
    
    
    int w = 40, h = 40;
    int chars = 1;
    std::string trainingImages[] = {"imgs/training/A.jpg","imgs/training/B.jpg","imgs/training/C.jpg",
                                    "imgs/training/D.jpg","imgs/training/E.jpg","imgs/training/F.jpg",
                                    "imgs/training/G.jpg","imgs/training/H.jpg",//"imgs/training/.jpg",
                                    "imgs/training/J.jpg","imgs/training/K.jpg","imgs/training/L.jpg",
                                    "imgs/training/M.jpg","imgs/training/N.jpg","imgs/training/O.jpg",
                                    "imgs/training/P.jpg","imgs/training/Q.jpg","imgs/training/R.jpg",
                                    "imgs/training/S.jpg","imgs/training/T.jpg","imgs/training/U.jpg",
                                    "imgs/training/V.jpg","imgs/training/W.jpg","imgs/training/X.jpg",
                                    "imgs/training/Y.jpg","imgs/training/Z.jpg"};
        
    ImageProcessor imgPrc;
    
    for (int i = 0; i < chars; i++) {
        
        imgPrc.initializeImage(trainingImages[i]);
        imgPrc.createCropedMatrix();
        imgPrc.resizeImage();
        imgPrc.skeletonize();
 
    }    
    
    
    
    return 0;
}

