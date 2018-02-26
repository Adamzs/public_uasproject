#ifndef TRACK_H
#define TRACK_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

class Track
{
    public:
        Rect rect;
        Point point;
        struct timeval tval;
        Track(Rect r, Point p, timeval time);
        bool isSizeSimilar(Rect r2);
        bool isCloseEnough(Point p2);
        double distanceBetween(const Point &a, const Point &b);

    protected:

    private:
};

#endif // TRACK_H
