#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QPainter>
#include <QMainWindow>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <opencv2/tracking.hpp>

using namespace std;
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    int nclicks, kernel, nZooms;
    Mat firstFrame, checkHighRes, highRes, countDrizzle, gaussCoeffsImg;
    QImage toShow;
    vector<Point> pointsForDrizzle;
    vector<Mat> blurredFrames, blurredBigPixels, finalFrames;
    Rect trackedRect;
    Rect2d rectDrizzle, rectDrizzle2, rectDrizzle3, bigRectDrizzle, bigRectDrizzle2, bigRectDrizzle3, bigRectCheck;
    Ptr<Tracker> tracker;
    Point startPoint, endPoint;
    bool lkChecked, farnebackChecked;
    int frameNumber;


    //GAUSSIAN SHARPENING
#define KERNEL_SIZE 11

    float sigma;  // Controlled by a slider
    float gauss[KERNEL_SIZE*KERNEL_SIZE];

    void ComputeGaussKernel(void);
    double gausserf(double x) {
        int sign=x<0?-1:1;
        x=fabs(x);
        double t=1.0f/(1.0f+0.3275911*x);
        return sign*(1.0f-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*exp(-x*x));
    }
    inline double defintgauss(double x, double m, double s) {return 0.5*(gausserf((x-m)/(1.4142135623731*s)));}




    bool eventFilter(QObject *Ob, QEvent *Ev);
    bool justOneTime;
    void buildTrackedRect();
    void zoomImage();
    Point obtainCorrelation(Mat templ, Mat img);

    void buildHighRes(Mat fatPixel, int yHigh, int xHigh, int yFat, int xFat, float weightedArea);

    void superresolution(Mat frame, vector<Point2f> fFatOffsets, vector<Point2i> iFatOffsets, int iOffsetX, int iOffsetY);
    void superresolution2(Mat frame, int iOffsetX, int iOffsetY, float fOffsetX, float fOffsetY, bool first);

    Point calculateCorrelation(Mat patch, Mat image);

private slots:
    void on_trackDrizzle_clicked();

    void on_showIt_clicked();

    void on_createSmallfromBig_clicked();

    void on_lk_stateChanged(int arg1);

    void on_farneback_stateChanged(int arg1);

    void on_sigmaValue_valueChanged(int value);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
