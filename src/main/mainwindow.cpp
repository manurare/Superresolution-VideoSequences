#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QEvent>
#include <QMouseEvent>
#include <qstatusbar.h>
#include <opencv2/core/core.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <QDebug>
#include <iostream>
#include <opencv2/tracking.hpp>
#include <QThread>

using namespace std;
using namespace cv;

bool MainWindow::eventFilter(QObject *Ob, QEvent *Ev)
{
    // Check captures event is a mouseclick
    if(Ev->type()==QEvent::MouseButtonPress) {
        // Check that it happened over the QLabel
        if(Ob==ui->blurred) {
            nclicks++;
            const QMouseEvent *me=static_cast<const QMouseEvent *>(Ev);
            // Click coordinates
            int y=me->y(), x=me->x();
            statusBar()->showMessage(QString::number(x)+":"+QString::number(y));
            // "True" if it has been managed correctly
            return true;
        }if(Ob==ui->reconstructed){
            const QMouseEvent *me=static_cast<const QMouseEvent *>(Ev);
            if(me->button() == Qt::MiddleButton){
                nZooms++;
                statusBar()->showMessage(QString::number(me->x())+":"+QString::number(me->y()));
                zoomImage();
            }
            if(me->button() == Qt::LeftButton){
                pointsForDrizzle.push_back(Point(me->x(),me->y()));
                if(pointsForDrizzle.size() == 4){
                    buildTrackedRect();
                }
            }
        }else return false;
    }
    return false;
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->blurred->installEventFilter(this);
    ui->reconstructed->installEventFilter(this);

    //Save blurred frames in a vector
    String path("/home/manuel/Desktop/TFG_ubuntu/Videos/blurredVideos/k11_pngFinal/*.png"); //select only png
    //String path("/home/manuel/Desktop/CLASE/TFG/Videos/CamaraRotating/*.jpg"); //select only bmp
    //String path("/home/manuel/Desktop/CLASE/TFG/Videos/sinteticosGood/*.png"); //select only bmp
    //String path("/home/manuel/Desktop/CLASE/TFG/Videos/10_05_2018/offsetsManuales/bigAndSmall/*.jpg"); //select only jpg
    vector<cv::String> fn;
    cv::glob(path, fn, true); // recurse
    for (size_t k = 0; k<fn.size(); ++k)
    {
        cv::Mat im = cv::imread(fn[k]);
        if (im.empty()) continue; //only proceed if sucsessful
        blurredFrames.push_back(im);
    }
    if(blurredFrames.empty()){
        qDebug("NO FRAMES");
        exit(0);
    }

    //INITIALIZATIONS
    ui->blurred->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    ui->reconstructed->setAlignment(Qt::AlignLeft | Qt::AlignTop);

    lkChecked = false;
    farnebackChecked = true;

    ui->lk->setChecked(lkChecked);
    ui->farneback->setChecked(farnebackChecked);

    //    //    mama k15
    //    startPoint.x = 78;
    //    startPoint.y = 8;
    //    endPoint.x = 618;
    //    endPoint.y = 428;
    //    kernel = 15;

    //    //    mama k10
    //    startPoint.x = 69;
    //    startPoint.y = 15;
    //    endPoint.x = 599;
    //    endPoint.y = 435;
    //    kernel = 10;

    //    //    mama k8
    //    startPoint.x = 74;
    //    startPoint.y = 15;
    //    endPoint.x = 594;
    //    endPoint.y = 423;
    //    kernel = 8;

    //    //    sinteticos
    //    startPoint.x = 120;
    //    startPoint.y = 60;
    //    endPoint.x = 540;
    //    endPoint.y = 420;
    //    kernel = KERNEL_SIZE;

    //    //    cameraRotating2 k15
    //    startPoint.x = 92;
    //    startPoint.y = 17;
    //    endPoint.x = 587;
    //    endPoint.y = 407;
    //    kernel = KERNEL_SIZE;

    //    //    cameraRotating2 k8
    //    startPoint.x = 80;
    //    startPoint.y = 17;
    //    endPoint.x = 560;
    //    endPoint.y = 409;
    //    kernel = 8;

    //    //    cameraRotating k10 png
    //    startPoint.x = 113;
    //    startPoint.y = 21;
    //    endPoint.x = 533;
    //    endPoint.y = 391;
    //    kernel = KERNEL_SIZE;

    //    cameraRotating k11 png
    startPoint.x = 73;
    startPoint.y = 17;
    endPoint.x = 568;
    endPoint.y = 380;
    kernel = KERNEL_SIZE;

    //    //    final k15 png
    //    startPoint.x = 82;
    //    startPoint.y = 20;
    //    endPoint.x = 547;
    //    endPoint.y = 380;
    //    kernel = KERNEL_SIZE;


    //    //    k10 png blurred
    //    startPoint.x = 97;
    //    startPoint.y = 21;
    //    endPoint.x = 537;
    //    endPoint.y = 391;
    //    kernel = KERNEL_SIZE;

    trackedRect.x = startPoint.x;
    trackedRect.y = startPoint.y;
    trackedRect.width = abs(startPoint.x-endPoint.x);
    trackedRect.height = abs(startPoint.y-endPoint.y);

    frameNumber = 0;

    pointsForDrizzle.clear();
    justOneTime = false;
    nclicks = nZooms = 0;
    blurredBigPixels.clear();
    sigma = ui->sigmaValue->value()*1.0f/100.0f;

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::buildTrackedRect(){
    if(!blurredBigPixels.empty() && pointsForDrizzle.size() == 4){
        for(size_t i = 0; i<pointsForDrizzle.size(); i++){
            cout<<pointsForDrizzle[i].x/pow(2,nZooms)<<", "<<pointsForDrizzle[i].y/pow(2,nZooms)<<endl;
        }
        rectDrizzle.x = floor(pointsForDrizzle[0].x/pow(2,nZooms)+0.5f);
        rectDrizzle.y = floor(pointsForDrizzle[0].y/pow(2,nZooms)+0.5f);
        rectDrizzle.width = abs(rectDrizzle.x-floor(pointsForDrizzle[3].x/pow(2,nZooms)+0.5f));
        rectDrizzle.height = abs(rectDrizzle.y-floor(pointsForDrizzle[3].y/pow(2,nZooms)+0.5f));

        rectDrizzle2.x = rectDrizzle3.x = rectDrizzle.x;
        rectDrizzle2.y = rectDrizzle3.y = rectDrizzle.y;
        rectDrizzle2.width = rectDrizzle3.width = rectDrizzle.width;
        rectDrizzle2.height = rectDrizzle3.height = rectDrizzle.height;

        bigRectDrizzle.x = bigRectDrizzle2.x = bigRectDrizzle3.x = rectDrizzle.x*kernel;
        bigRectDrizzle.y = bigRectDrizzle2.y = bigRectDrizzle3.y = rectDrizzle.y*kernel;
        bigRectDrizzle.width = bigRectDrizzle2.width = bigRectDrizzle3.width = rectDrizzle.width*kernel;
        bigRectDrizzle.height = bigRectDrizzle2.height = bigRectDrizzle3.height = rectDrizzle.height*kernel;

        cout<<endl;
        cout<<rectDrizzle.x<<", "<<rectDrizzle.y<<endl;
        cout<<rectDrizzle.width<<", "<<rectDrizzle.height<<endl;

        cout<<endl;

        Mat matToShow = blurredBigPixels[0].clone();
        rectangle(matToShow,rectDrizzle,Scalar(255,255,255),1);
        cvtColor(matToShow,matToShow,COLOR_BGR2RGB);
        toShow = QImage(matToShow.data, matToShow.cols, matToShow.rows, matToShow.step, QImage::Format_RGB888);
        ui->reconstructed->setScaledContents(true);
        ui->reconstructed->setPixmap(QPixmap::fromImage(toShow));
    }
}

void MainWindow::on_trackDrizzle_clicked()
{
    if(!blurredBigPixels.empty() && !rectDrizzle.empty()){
        if(ui->lk->isChecked() || ui->farneback->isChecked()){

            Mat current, currentGray, currentBig;
            Mat previousBig = blurredFrames[0].clone();
            Mat previous = blurredBigPixels[0].clone();
            Mat previous2, previous3;

            Mat previousGray, previous2Gray, previous3Gray;
            cvtColor(previous,previousGray,COLOR_BGR2GRAY);

            Mat previousBigGray, currentBigGray;
            cvtColor(previousBig, previousBigGray, COLOR_BGR2GRAY);

            vector<uchar> status;
            vector<float> err;
            vector<Point2i> iFatPixelsOffset;
            vector<Point2f> incOffsets, incOffsets2PrevCurr, fFatPixelsOffset, incOffsets3PrevCurr;
            vector<Point2f> prevPoints, currPoints;

            float sumIncOffsetX = 0.0f, sumIncOffsetY = 0.0f;
            float incOffsetX = 0.0f, incOffsetY = 0.0f;

            highRes = Mat::zeros(trackedRect.height+250, trackedRect.width+250,CV_32FC3);

            ComputeGaussKernel();
            for(int i=0; i<KERNEL_SIZE*KERNEL_SIZE; i++){
                cout<<gauss[i]<<endl;
            }
            cout<<endl<<endl;

            countDrizzle = Mat::zeros(highRes.rows,highRes.cols, CV_32FC1);
            checkHighRes = Mat::zeros(highRes.rows, highRes.cols, CV_8UC3);

            for(int i=0; i<previous.rows; i++) for(int j=0; j<previous.cols; j++){
                fFatPixelsOffset.push_back(Point2f(0.0f,0.0f));
                iFatPixelsOffset.push_back(Point2i(0,0));
            }
            for(int i=rectDrizzle.y; i<rectDrizzle.y+rectDrizzle.height; i++){
                for(int j=rectDrizzle.x; j<rectDrizzle.x+rectDrizzle.width;j++){
                    prevPoints.push_back(Point2f(j,i));
                }
            }

            superresolution(previousBig(trackedRect),fFatPixelsOffset,iFatPixelsOffset,0,0);

            //START OPTICAL FLOW
            for(size_t k=1; k<blurredBigPixels.size(); k++){
                frameNumber++;
                cout<<"FRAME= "<<k<<endl;
                fFatPixelsOffset.clear();
                iFatPixelsOffset.clear();

                sumIncOffsetX = sumIncOffsetY = incOffsetX = incOffsetY = 0.0f;

                float sumIncOffsetXprev2Curr = 0.0f, sumIncOffsetYprev2Curr = 0.0f, incOffsetXprev2Curr = 0.0f, incOffsetYprev2Curr = 0.0f;
                float sumIncOffsetXprev3Curr = 0.0f, sumIncOffsetYprev3Curr = 0.0f, incOffsetXprev3Curr = 0.0f, incOffsetYprev3Curr = 0.0f;

                if(k==2){
                    previous2 = blurredBigPixels[0].clone();
                    cvtColor(previous2,previous2Gray,COLOR_BGR2GRAY);
                }

                if(k==3){
                    previous3 = blurredBigPixels[0].clone();
                    cvtColor(previous3,previous3Gray,COLOR_BGR2GRAY);
                }

                currentBig = blurredFrames[k].clone();
                current = blurredBigPixels[k].clone();
                cvtColor(current,currentGray,COLOR_BGR2GRAY);
                cvtColor(currentBig, currentBigGray, COLOR_BGR2GRAY);

                if(ui->lk->isChecked()){
                    //L-K SPARSE OPTICAL FLOW
                    if(k==1) ui->status->appendPlainText("L-K");

                    calcOpticalFlowPyrLK(previousGray,currentGray,prevPoints,currPoints,status,err,Size(21,21));

                    for( size_t i = 0; i < prevPoints.size(); i++ )
                    {
                        sumIncOffsetX += (currPoints[i].x-prevPoints[i].x);
                        sumIncOffsetY += (currPoints[i].y-prevPoints[i].y);

                        fFatPixelsOffset[i].x = (currPoints[i].x-prevPoints[i].x)*kernel;
                        fFatPixelsOffset[i].y = (currPoints[i].y-prevPoints[i].y)*kernel;

                        iFatPixelsOffset[i].x = (currPoints[i].x-prevPoints[i].x)*kernel;
                        iFatPixelsOffset[i].y = (currPoints[i].y-prevPoints[i].y)*kernel;

                    }
                    incOffsetX = sumIncOffsetX*kernel/prevPoints.size();
                    incOffsetY = sumIncOffsetY*kernel/prevPoints.size();

                }else{

                    //FARNEBACK DENSE OPTICAL FLOW
                    if(k==1) ui->status->appendPlainText("FARNEBACK");

                    Mat flow, faceRect, faceRectPrev2Curr, faceRectPrev3Curr;

                    calcOpticalFlowFarneback(previousGray,currentGray,flow,0.5,1,12,2,8,1.2,0);
                    faceRect = flow.clone()(rectDrizzle);

                    //                    /********************************************DRAW OPTICAL FLOW***********************************************/

                    //                    Mat arrowOpticalFlow = currentBig(trackedRect).clone();
                    //                    for (int y = 0; y < arrowOpticalFlow.rows; y+=kernel) {
                    //                        for (int x = 0; x < arrowOpticalFlow.cols; x+=kernel)
                    //                        {
                    //                            // get the flow from y, x position * 10 for better visibility
                    //                            const Point2f flowatxy = flow.at<Point2f>(y/kernel, x/kernel);
                    //                            // draw line at flow direction
                    //                            line(arrowOpticalFlow, Point(x, y), Point(cvRound(x + flowatxy.x*kernel*2), cvRound(y + flowatxy.y*kernel*2)), Scalar(0,0,0));
                    //                            // draw initial point
                    //                            circle(arrowOpticalFlow, Point(x, y), 1, Scalar(0, 0, 0), -1);
                    //                        }
                    //                    }
                    //                    namedWindow("prew", WINDOW_NORMAL);
                    //                    imshow("prew", arrowOpticalFlow);
                    //                    if(frameNumber<10){
                    //                        imwrite("/home/manuel/Desktop/CLASE/TFG/Presentacion/OpticalFlow/00"+to_string(frameNumber)+".jpg",arrowOpticalFlow);
                    //                    }
                    //                    if(frameNumber>=10 && frameNumber<100){
                    //                        imwrite("/home/manuel/Desktop/CLASE/TFG/Presentacion/OpticalFlow/0"+to_string(frameNumber)+".jpg",arrowOpticalFlow);
                    //                    }
                    //                    if(frameNumber>=100){
                    //                        imwrite("/home/manuel/Desktop/CLASE/TFG/Presentacion/OpticalFlow/"+to_string(frameNumber)+".jpg",arrowOpticalFlow);
                    //                    }

                    //                    waitKey(0);

                    /********************************************DRAW OPTICAL FLOW***********************************************/

                    if(k>=2){
                        calcOpticalFlowFarneback(previous2Gray,currentGray,flow,0.5,1,12,2,8,1.2,0);
                        faceRectPrev2Curr = flow.clone()(rectDrizzle2);
                    }

                    if(k>=3){
                        calcOpticalFlowFarneback(previous3Gray,currentGray,flow,0.5,1,12,2,8,1.2,0);
                        faceRectPrev3Curr = flow.clone()(rectDrizzle3);
                    }

                    for(int i = 0; i < faceRect.rows; i++) for( int j = 0; j < faceRect.cols; j++)
                    {
                        sumIncOffsetX += faceRect.at<Vec2f>(i,j)[0];
                        sumIncOffsetY += faceRect.at<Vec2f>(i,j)[1];

                        if(k>=2){
                            sumIncOffsetXprev2Curr += faceRectPrev2Curr.at<Vec2f>(i,j)[0];
                            sumIncOffsetYprev2Curr += faceRectPrev2Curr.at<Vec2f>(i,j)[1];
                        }

                        if(k>=3){
                            sumIncOffsetXprev3Curr += faceRectPrev3Curr.at<Vec2f>(i,j)[0];
                            sumIncOffsetYprev3Curr += faceRectPrev3Curr.at<Vec2f>(i,j)[1];
                        }
                    }
                    incOffsetX = sumIncOffsetX*kernel/(faceRect.rows*faceRect.cols);
                    incOffsetY = sumIncOffsetY*kernel/(faceRect.rows*faceRect.cols);

                    if(k>=2){
                        incOffsetXprev2Curr = sumIncOffsetXprev2Curr*kernel/(faceRect.rows*faceRect.cols);
                        incOffsetYprev2Curr = sumIncOffsetYprev2Curr*kernel/(faceRect.rows*faceRect.cols);
                    }else{
                        incOffsetXprev2Curr = 0;
                        incOffsetYprev2Curr = 0;
                    }

                    if(k>=3){
                        incOffsetXprev3Curr = sumIncOffsetXprev3Curr*kernel/(faceRect.rows*faceRect.cols);
                        incOffsetYprev3Curr = sumIncOffsetYprev3Curr*kernel/(faceRect.rows*faceRect.cols);
                    }else{
                        incOffsetXprev3Curr = 0;
                        incOffsetYprev3Curr = 0;
                    }
                    for(int i=0; i<flow.rows; i++) for(int j=0; j<flow.cols; j++){
                        fFatPixelsOffset.push_back(Point2f(flow.at<Vec2f>(i,j)[0]*kernel, flow.at<Vec2f>(i,j)[1]*kernel));

                        iFatPixelsOffset.push_back(Point2i(floor(flow.at<Vec2f>(i,j)[0]*kernel), floor(flow.at<Vec2f>(i,j)[1]*kernel)));
                    }

                }

                //////////////////////////////////////////////////////

                incOffsets.push_back(Point2f(incOffsetX,incOffsetY));

                incOffsets2PrevCurr.push_back(Point2f(incOffsetXprev2Curr,incOffsetYprev2Curr));

                incOffsets3PrevCurr.push_back(Point2f(incOffsetXprev3Curr,incOffsetYprev3Curr));

                float totalOffsetX = 0.0f, totalOffsetY = 0.0f, totalOffsetXPrev2Curr = 0.0f, totalOffsetYPrev2Curr= 0.0f;
                float totalOffsetXPrev3Curr = 0.0f, totalOffsetYPrev3Curr= 0.0f;

                float sumFinalX = 0.0f, sumFinalY=0.0f;
                int methods = 0;

                //Incremental f-1
                for(size_t i=0; i<incOffsets.size(); i++){
                    totalOffsetX += incOffsets[i].x;
                    totalOffsetY += incOffsets[i].y;
                }
                sumFinalX += totalOffsetX;
                sumFinalY += totalOffsetY;
                //cout<<totalOffsetX<<", "<<totalOffsetY<<endl;
                methods++;

                //incrementals f-2 only if k is a multiple of 2
                if(k%2==0){
                    for(size_t i=0; i<incOffsets2PrevCurr.size(); i++){
                        if(i>=1 && (i+1)%2==0){
                            totalOffsetXPrev2Curr += incOffsets2PrevCurr[i].x;
                            totalOffsetYPrev2Curr += incOffsets2PrevCurr[i].y;
                        }else{
                            totalOffsetXPrev2Curr += 0;
                            totalOffsetYPrev2Curr += 0;
                        }
                    }
                    sumFinalX += totalOffsetXPrev2Curr;
                    sumFinalY += totalOffsetYPrev2Curr;
                    methods++;
                }
                //cout<<totalOffsetXPrev2Curr<<", "<<totalOffsetYPrev2Curr<<endl;

                //incrementals f-3 only if k is a multiple of 3
                if(k%3==0){
                    for(size_t i=0; i<incOffsets3PrevCurr.size(); i++){
                        if((i+1)%3==0){
                            totalOffsetXPrev3Curr += incOffsets3PrevCurr[i].x;
                            totalOffsetYPrev3Curr += incOffsets3PrevCurr[i].y;
                        }else{
                            totalOffsetXPrev3Curr += 0;
                            totalOffsetYPrev3Curr += 0;
                        }
                    }
                    sumFinalX += totalOffsetXPrev3Curr;
                    sumFinalY += totalOffsetYPrev3Curr;
                    methods++;
                }
                //cout<<totalOffsetXPrev3Curr<<", "<<totalOffsetYPrev3Curr<<endl;

                //2 by 2 + first one from f-1
                float newMethodX = 0.0f, newMethodY = 0.0f;
                if(k%2!=0){
                    for(size_t i=0; i<incOffsets2PrevCurr.size(); i++){
                        if((i+1)%2!=0){
                            newMethodX += incOffsets2PrevCurr[i].x;
                            newMethodY += incOffsets2PrevCurr[i].y;
                        }
                    }
                    newMethodX += incOffsets[0].x;
                    newMethodY += incOffsets[0].y;
                    sumFinalX += newMethodX;
                    sumFinalY += newMethodY;
                    methods++;
                }
                //cout<<newMethodX<<", "<<newMethodY<<endl;

                //3 by 3 + first one of f-1
                newMethodX = newMethodY = 0.0f;
                if(k%3==1){
                    for(size_t i=0; i<incOffsets3PrevCurr.size(); i++){
                        if((i+1)%3 == 1){
                            newMethodX += incOffsets3PrevCurr[i].x;
                            newMethodY += incOffsets3PrevCurr[i].y;
                        }
                    }
                    newMethodX += incOffsets[0].x;
                    newMethodY += incOffsets[0].y;
                    sumFinalX += newMethodX;
                    sumFinalY += newMethodY;
                    methods++;
                }
                //cout<<newMethodX<<", "<<newMethodY<<endl;

                //3 by 3 + first and second one of f-1
                newMethodX = newMethodY = 0.0f;
                if(k%3==2){
                    for(size_t i=0; i<incOffsets3PrevCurr.size(); i++){
                        if((i+1)%3 == 2){
                            newMethodX += incOffsets3PrevCurr[i].x;
                            newMethodY += incOffsets3PrevCurr[i].y;
                        }
                    }
                    newMethodX += incOffsets[0].x;
                    newMethodY += incOffsets[0].y;
                    newMethodX += incOffsets[1].x;
                    newMethodY += incOffsets[1].y;
                    sumFinalX += newMethodX;
                    sumFinalY += newMethodY;
                    methods++;
                }
                //cout<<newMethodX<<", "<<newMethodY<<endl;

                //1 by 1 until two before k
                newMethodX = newMethodY = 0.0f;
                if(k>=3){
                    for(size_t i=0; i<incOffsets.size(); i++){
                        if((i+1)<=k-2){
                            newMethodX += incOffsets[i].x;
                            newMethodY += incOffsets[i].y;
                        }
                    }
                    newMethodX += incOffsets2PrevCurr[k-1].x;
                    newMethodY += incOffsets2PrevCurr[k-1].y;
                    sumFinalX += newMethodX;
                    sumFinalY += newMethodY;
                    methods++;
                }
                // cout<<newMethodX<<", "<<newMethodY<<endl;

                //1 by 1 until three before k
                newMethodX = newMethodY = 0.0f;
                if(k>=4){
                    for(size_t i=0; i<incOffsets.size(); i++){
                        if((i+1)<=k-3){
                            newMethodX += incOffsets[i].x;
                            newMethodY += incOffsets[i].y;
                        }
                    }
                    newMethodX += incOffsets3PrevCurr[k-1].x;
                    newMethodY += incOffsets3PrevCurr[k-1].y;
                    sumFinalX += newMethodX;
                    sumFinalY += newMethodY;
                    methods++;
                }
                //cout<<newMethodX<<", "<<newMethodY<<endl;

                //                cout<<methods<<endl;
                //                cout<<"FINAL"<<endl;
                float fFINALX = sumFinalX/methods;
                float fFINALY = sumFinalY/methods;
                int iFINALX = floor(fFINALX + 0.5f);
                int iFINALY = floor(fFINALY + 0.5f);

                int finalTotalOffsetX = floor(totalOffsetX+0.5f);
                int finalTotalOffsetY = floor(totalOffsetY+0.5f);
                //cout<<finalTotalOffsetX<<", "<<finalTotalOffsetY<<endl;

                int finalTotalOffsetXPrev2Curr = floor(totalOffsetXPrev2Curr+0.5f);
                int finalTotalOffsetYPrev2Curr = floor(totalOffsetYPrev2Curr+0.5f);

                int finalTotalOffsetXPrev3Curr = floor(totalOffsetXPrev3Curr+0.5f);
                int finalTotalOffsetYPrev3Curr = floor(totalOffsetYPrev3Curr+0.5f);

                rectDrizzle.x += incOffsetX/kernel;
                rectDrizzle.y += incOffsetY/kernel;
                bigRectDrizzle.x += incOffsetX;
                bigRectDrizzle.y += incOffsetY;

                if(k%2==0){
                    rectDrizzle2.x += incOffsetXprev2Curr/kernel;
                    rectDrizzle2.y += incOffsetYprev2Curr/kernel;
                    bigRectDrizzle2.x += incOffsetXprev2Curr;
                    bigRectDrizzle2.y += incOffsetYprev2Curr;
                }

                if(k%3==0){
                    rectDrizzle3.x += incOffsetXprev3Curr/kernel;
                    rectDrizzle3.y += incOffsetYprev3Curr/kernel;
                    bigRectDrizzle3.x += incOffsetXprev3Curr;
                    bigRectDrizzle3.y += incOffsetYprev3Curr;
                }

                Mat auxRect = currentBig.clone()(trackedRect);
                rectangle(auxRect,bigRectDrizzle,Scalar(255,255,255),1);
                namedWindow("rect",WINDOW_NORMAL);
                imshow("rect",auxRect);
                waitKey(0);
                if(frameNumber<10){
                    imwrite("/home/manuel/Desktop/CLASE/TFG/Presentacion/FaceTracking/00"+to_string(frameNumber)+".jpg",auxRect);
                }
                if(frameNumber>=10 && frameNumber<100){
                    imwrite("/home/manuel/Desktop/CLASE/TFG/Presentacion/FaceTracking/0"+to_string(frameNumber)+".jpg",auxRect);
                }
                if(frameNumber>=100){
                    imwrite("/home/manuel/Desktop/CLASE/TFG/Presentacion/FaceTracking/"+to_string(frameNumber)+".jpg",auxRect);
                }


                if(k>=3){
                    ui->status->appendPlainText("Incrementales");
                    ui->status->appendPlainText(QString::number(incOffsetX)+", "+QString::number(incOffsetY));
                    ui->status->appendPlainText(QString::number(incOffsetXprev2Curr)+", "+QString::number(incOffsetYprev2Curr));
                    ui->status->appendPlainText(QString::number(incOffsetXprev3Curr)+", "+QString::number(incOffsetYprev3Curr));
                    ui->status->appendPlainText("Totales float");
                    ui->status->appendPlainText(QString::number(totalOffsetX)+", "+QString::number(totalOffsetY));
                    ui->status->appendPlainText(QString::number(totalOffsetXPrev2Curr)+", "+QString::number(totalOffsetYPrev2Curr));
                    ui->status->appendPlainText(QString::number(totalOffsetXPrev3Curr)+", "+QString::number(totalOffsetYPrev3Curr));
                    ui->status->appendPlainText("Totales int");
                    ui->status->appendPlainText(QString::number(finalTotalOffsetX)+", "+QString::number(finalTotalOffsetY));
                    ui->status->appendPlainText(QString::number(finalTotalOffsetXPrev2Curr)+", "+QString::number(finalTotalOffsetYPrev2Curr));
                    ui->status->appendPlainText(QString::number(finalTotalOffsetXPrev3Curr)+", "+QString::number(finalTotalOffsetYPrev3Curr));
                    ui->status->appendPlainText(QString::number(iFINALX)+", "+QString::number(iFINALY));
                }else if(k>=2){
                    ui->status->appendPlainText("Incrementales");
                    ui->status->appendPlainText(QString::number(incOffsetX)+", "+QString::number(incOffsetY));
                    ui->status->appendPlainText(QString::number(incOffsetXprev2Curr)+", "+QString::number(incOffsetYprev2Curr));
                    ui->status->appendPlainText("Totales float");
                    ui->status->appendPlainText(QString::number(totalOffsetX)+", "+QString::number(totalOffsetY));
                    ui->status->appendPlainText(QString::number(totalOffsetXPrev2Curr)+", "+QString::number(totalOffsetYPrev2Curr));
                    ui->status->appendPlainText("Totales int");
                    ui->status->appendPlainText(QString::number(finalTotalOffsetX)+", "+QString::number(finalTotalOffsetY));
                    ui->status->appendPlainText(QString::number(finalTotalOffsetXPrev2Curr)+", "+QString::number(finalTotalOffsetYPrev2Curr));
                    ui->status->appendPlainText(QString::number(iFINALX)+", "+QString::number(iFINALY));
                }else{
                    ui->status->appendPlainText("Incrementales");
                    ui->status->appendPlainText(QString::number(incOffsetX)+", "+QString::number(incOffsetY));
                    ui->status->appendPlainText("Totales float");
                    ui->status->appendPlainText(QString::number(totalOffsetX)+", "+QString::number(totalOffsetY));
                    ui->status->appendPlainText("Totales int");
                    ui->status->appendPlainText(QString::number(finalTotalOffsetX)+", "+QString::number(finalTotalOffsetY));
                    ui->status->appendPlainText(QString::number(iFINALX)+", "+QString::number(iFINALY));
                }
                superresolution(currentBig(trackedRect),fFatPixelsOffset, iFatPixelsOffset, iFINALX, iFINALY);

                //superresolution2(currentBig(trackedRect),iFINALX, iFINALY, fFINALX, fFINALY, false);
                if(k>=3){
                    prevPoints = currPoints;
                    previous3Gray = previous2Gray.clone();
                    previous2Gray = previousGray.clone();
                    previousGray = currentGray.clone();
                    previousBigGray = currentBigGray.clone();
                }
                else if(k>=2){
                    prevPoints = currPoints;
                    previous2Gray = previousGray.clone();
                    previousGray = currentGray.clone();
                    previousBigGray = currentBigGray.clone();
                }else{
                    prevPoints = currPoints;
                    previous = current.clone();
                    previousGray = currentGray.clone();
                    previousBigGray = currentBigGray.clone();
                }

            }

            Mat finalShow = Mat::zeros(highRes.rows,highRes.cols,CV_8UC3);

            for(int i=0; i<highRes.rows; i++) for(int j=0; j<highRes.cols; j++){
                //if((i*kernel+j)%KERNEL_SIZE/2 == 0)  cout<<highRes.at<Vec3f>(i,j)[0]<<", "<<countDrizzle.at<float>(i,j)<<endl;
                float value = countDrizzle.at<float>(i,j);

                if(value == 0.0f) value = 1.0f;

                if(!ui->CBInhibit->isChecked()){
                    finalShow.at<Vec3b>(i,j)[0] = (highRes.at<Vec3f>(i,j)[0] / value);
                    finalShow.at<Vec3b>(i,j)[1] = (highRes.at<Vec3f>(i,j)[1] / value);
                    finalShow.at<Vec3b>(i,j)[2] = (highRes.at<Vec3f>(i,j)[2] / value);
                }else{
                    float pixR = 0.0f, pixG = 0.0f, pixB = 0.0f;
                    pixB = highRes.at<Vec3f>(i,j)[0] / value;
                    pixG = highRes.at<Vec3f>(i,j)[1] / value;
                    pixR = highRes.at<Vec3f>(i,j)[2] / value;

                    if(pixB<0.0f){
                        pixB = fabs(pixB);
                        if(pixB > 255) pixB = 255;
                        finalShow.at<Vec3b>(i,j)[0] = pixB;
                    }else finalShow.at<Vec3b>(i,j)[0] = pixB;

                    if(pixG<0.0f){
                        pixG = fabs(pixG);
                        if(pixG > 255) pixG = 255;
                        finalShow.at<Vec3b>(i,j)[1] = pixG;
                    }else finalShow.at<Vec3b>(i,j)[1] = pixG;

                    if(pixR<0.0f){
                        pixR = fabs(pixR);
                        if(pixR > 255) pixR = 255;
                        finalShow.at<Vec3b>(i,j)[2] = pixR;
                    }else finalShow.at<Vec3b>(i,j)[2] = pixR;


                    if(pixG<0.0f) finalShow.at<Vec3b>(i,j)[1] = fabs(pixG)==0?0:(fabs(pixG)==255?255:fabs(pixG));
                    else finalShow.at<Vec3b>(i,j)[1] = pixG;
                    if(pixR<0.0f) finalShow.at<Vec3b>(i,j)[2] = fabs(pixR)==0?0:(fabs(pixR)==255?255:fabs(pixR));
                    else finalShow.at<Vec3b>(i,j)[2] = pixR;
                }
            }

            Mat finalRGB;
            Mat noiseRemoved;
            fastNlMeansDenoisingColored(finalShow,noiseRemoved,5,11,7,21);
            cvtColor(finalShow,finalRGB,COLOR_BGR2RGB);
            toShow = QImage(finalRGB.data, finalRGB.cols, finalRGB.rows, finalRGB.step, QImage::Format_RGB888);
            ui->reconstructed->setPixmap(QPixmap::fromImage(toShow));
            imwrite("/home/manuel/Desktop/CLASE/TFG/Videos/TestImages/final.png",finalShow(Rect(Point2i(125,125),trackedRect.size())));
            namedWindow("FINAL", WINDOW_NORMAL);
            imshow("FINAL",finalShow);
            namedWindow("Noise Removed", WINDOW_NORMAL);
            imshow("Noise Removed",noiseRemoved);
            waitKey(0);
        }else{
            qWarning()<<"Choose an optical flow method";
        }
    }else{
        qWarning()<<"Need to define an area to work";
    }
}

void MainWindow::on_showIt_clicked()
{
    firstFrame = blurredFrames[0].clone();
    cvtColor(firstFrame,firstFrame,CV_BGR2RGB);
    toShow = QImage(firstFrame.data, firstFrame.cols, firstFrame.rows, firstFrame.step, QImage::Format_RGB888);
    ui->blurred->setPixmap(QPixmap::fromImage(toShow));
}

void MainWindow::superresolution(Mat frame, vector<Point2f> fFatOffsets, vector<Point2i> iFatOffsets, int iOffsetX, int iOffsetY){
    vector<Rect> vectorFatPixels;

    for(int i=0; i<frame.rows; i+=kernel) for(int j=0; j<frame.cols; j+=kernel){
        Rect aux;
        aux.x = j;
        aux.y = i;
        aux.width = aux.height = kernel;
        vectorFatPixels.push_back(aux);
    }

    Mat check = Mat::zeros(highRes.rows, highRes.cols, CV_8UC3);

    for(size_t k = 0; k<vectorFatPixels.size(); k++){
        Mat fatPixel = frame(vectorFatPixels[k]).clone();
        float shiftX = 0.0f;
        float shiftY = 0.0f;
        if(shiftX<0.0f || shiftY<0.0f) cout<<shiftX<<", "<<shiftY<<endl;
        float value;
        int pixRFat=0, pixBFat=0, pixGFat=0;
        Mat pixelGauss = Mat::zeros(fatPixel.rows, fatPixel.cols, CV_32FC3);
        for(int i=0; i<pixelGauss.rows; i++) for(int j=0; j<pixelGauss.cols; j++){
            pixRFat = fatPixel.at<Vec3b>(i,j)[0];
            pixGFat = fatPixel.at<Vec3b>(i,j)[1];
            pixBFat = fatPixel.at<Vec3b>(i,j)[2];

            pixelGauss.at<Vec3f>(i,j)[0] = pixRFat*gauss[i*kernel+j];
            pixelGauss.at<Vec3f>(i,j)[1] = pixGFat*gauss[i*kernel+j];
            pixelGauss.at<Vec3f>(i,j)[2] = pixBFat*gauss[i*kernel+j];
        }

        int startX = (highRes.cols-trackedRect.width)/2 + vectorFatPixels[k].x - iOffsetX;
        int startY = (highRes.rows-trackedRect.height)/2 + vectorFatPixels[k].y - iOffsetY;
        int endX = startX+fatPixel.cols;
        int endY = startY+fatPixel.rows;

        Mat paintFat = Mat::zeros(KERNEL_SIZE+1, KERNEL_SIZE+1, CV_8UC3);

        for(int i=startY; i<endY; i++) for(int j=startX; j<endX; j++){
            for(int y=i; y<i+2; y++) for(int x=j; x<j+2; x++){
                if((y-i)*2+(x-j) == 0){
                    value = (1-shiftX)*(1-shiftY);
                    highRes.at<Vec3f>(y,x)[0] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[0]*value);
                    highRes.at<Vec3f>(y,x)[1] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[1]*value);
                    highRes.at<Vec3f>(y,x)[2] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[2]*value);
                    countDrizzle.at<float>(y,x) += (gauss[(i-startY)*kernel+(j-startX)]*value);
                }else if((y-i)*2+(x-j) == 1){
                    value = (shiftX)*(1-shiftY);
                    highRes.at<Vec3f>(y,x)[0] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[0]*value);
                    highRes.at<Vec3f>(y,x)[1] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[1]*value);
                    highRes.at<Vec3f>(y,x)[2] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[2]*value);
                    countDrizzle.at<float>(y,x) += (gauss[(i-startY)*kernel+(j-startX)]*value);
                }else if((y-i)*2+(x-j) == 2){
                    value = (1-shiftX)*(shiftY);
                    highRes.at<Vec3f>(y,x)[0] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[0]*value);
                    highRes.at<Vec3f>(y,x)[1] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[1]*value);
                    highRes.at<Vec3f>(y,x)[2] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[2]*value);
                    countDrizzle.at<float>(y,x) += (gauss[(i-startY)*kernel+(j-startX)]*value);
                }else{
                    value = (shiftX)*(shiftY);
                    highRes.at<Vec3f>(y,x)[0] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[0]*value);
                    highRes.at<Vec3f>(y,x)[1] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[1]*value);
                    highRes.at<Vec3f>(y,x)[2] += (pixelGauss.at<Vec3f>(i-startY,j-startX)[2]*value);
                    countDrizzle.at<float>(y,x) += (gauss[(i-startY)*kernel+(j-startX)]*value);
                }
            }
        }
    }
}

void MainWindow::ComputeGaussKernel(void) {

    const int K=KERNEL_SIZE;
    float *coeff=new float[K];
    float mu=0.0f;
    float xi=-K/2.0f, dx=1.0f;
    float sum=0;
    double oint=defintgauss(xi,mu,sigma);
    for(int k=0;k<K;++k) {
        double nint=defintgauss(xi+=dx,mu,sigma);
        sum+=(coeff[k]=nint-oint);
        oint=nint;
    }
    for(int k=0;k<K;k++) coeff[k]/=sum;
    for(int u=0;u<K;u++) for(int v=0;v<K;v++)
        gauss[u*K+v]=coeff[u]*coeff[v];
    delete[] coeff;

    // Lateral inhibition ("sharpening style" kernel)
    if(ui->CBInhibit->isChecked()) {
        const int K2=K*K;
        sum=0; for(int i=0;i<K2;i++) sum+=(gauss[i]*=-1);
        sum-=gauss[K2/2];  // Center coefficient
        gauss[K2/2]=1-sum; // Normalized kernel
    }

}

void MainWindow::on_createSmallfromBig_clicked()
{
    pointsForDrizzle.clear();
    nZooms = 0;
    ui->reconstructed->setScaledContents(false);
    if(trackedRect.width > 0 && trackedRect.height > 0){
        if(blurredBigPixels.empty()){
            Mat smallPixels = Mat::zeros(trackedRect.height/kernel, trackedRect.width/kernel, CV_8UC3);
            Mat tmpRect;
            for(size_t k=0; k<blurredFrames.size(); k++){
                tmpRect = blurredFrames[k].clone();
                tmpRect = tmpRect(trackedRect);
                for(int i=0; i<tmpRect.rows; i+=kernel) for(int j=0; j<tmpRect.cols; j+=kernel){
                    smallPixels.at<Vec3b>(i/kernel,j/kernel)[0] = tmpRect.at<Vec3b>(i,j)[0];
                    smallPixels.at<Vec3b>(i/kernel,j/kernel)[1] = tmpRect.at<Vec3b>(i,j)[1];
                    smallPixels.at<Vec3b>(i/kernel,j/kernel)[2] = tmpRect.at<Vec3b>(i,j)[2];
                }
                Mat toVector = smallPixels.clone();
                blurredBigPixels.push_back(toVector);
            }
            ui->status->appendPlainText("Conversion from big chunks to small chunks DONE!");
            Mat matToShow;
            cvtColor(blurredBigPixels[0],matToShow,COLOR_BGR2RGB);
            toShow = QImage(matToShow.data, matToShow.cols, matToShow.rows, matToShow.step, QImage::Format_RGB888);
            ui->reconstructed->setPixmap(QPixmap::fromImage(toShow));
            return;
        }else{
            ui->status->appendPlainText("Conversion from big chunks to small chunks DONE!");
            Mat matToShow;
            cvtColor(blurredBigPixels[0],matToShow,COLOR_BGR2RGB);
            toShow = QImage(matToShow.data, matToShow.cols, matToShow.rows, matToShow.step, QImage::Format_RGB888);
            ui->reconstructed->setPixmap(QPixmap::fromImage(toShow));
            return;
        }
    }else{
        qWarning()<<"There is no measurements to proceed";
    }
}

void MainWindow::zoomImage(){
    if(ui->reconstructed->pixmap() != 0){
        QImage onLabel = ui->reconstructed->pixmap()->toImage();
        QImage tmpScaled;
        if(2*onLabel.width() < ui->reconstructed->width() && 2*onLabel.height()<ui->reconstructed->height()){
            tmpScaled = onLabel.scaled(2*onLabel.width(), 2*onLabel.height());
            ui->reconstructed->setPixmap(QPixmap::fromImage(tmpScaled));
        }else{
            nZooms--;
        }
    }
}

void MainWindow::on_lk_stateChanged(int arg1)
{
    if(arg1 == 0){
        lkChecked = false;
        ui->lk->setChecked(lkChecked);
    }
    if(arg1 == 2){
        lkChecked = true;
        farnebackChecked = false;
        ui->farneback->setChecked(farnebackChecked);
        ui->lk->setChecked(lkChecked);
    }
}

void MainWindow::on_farneback_stateChanged(int arg1)
{
    if(arg1 == 0){
        farnebackChecked = false;
        ui->farneback->setChecked(farnebackChecked);
    }
    if(arg1 == 2){
        farnebackChecked = true;
        lkChecked = false;
        ui->farneback->setChecked(farnebackChecked);
        ui->lk->setChecked(lkChecked);
    }
}

void MainWindow::on_sigmaValue_valueChanged(int value)
{
    sigma = value/100.0f*1.0f;
    ui->status->appendPlainText("sigma= "+QString::number(sigma));
}
