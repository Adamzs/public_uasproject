#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#include <algorithm>

#include "Track.hpp"

using namespace cv;
using namespace std;

Track::Track(Rect r, Point p, timeval time)
{
    rect = r;
    point = p;
    tval = time;
}

bool Track::isSizeSimilar(Rect r2)
{
    return std::abs(1 - (rect.area() / r2.area())) < 0.3;
}

bool Track::isCloseEnough(Point p2)
{
    cout << rect.area() << "\n";
    return distanceBetween(point, p2) < (rect.area() / 200);
}

double Track::distanceBetween(const Point &a, const Point &b) {
    double xDiff = a.x - b.x;
    double yDiff = a.y - b.y;

    return sqrt((xDiff * xDiff) + (yDiff * yDiff));
}

