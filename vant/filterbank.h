#ifndef FILTERBANK_H
#define FILTERBANK_H

#include <opencv2/opencv.hpp>

class FilterBank
{
    public:
        FilterBank(int size);

        int size;
        std::string name;
        std::vector<cv::Mat> filters;
        std::vector<cv::Mat> norm_filters;
        std::vector<cv::Mat> samples;
        std::vector<cv::Mat> norm_samples;
        std::vector<std::vector<float>> filter_responses;

        // Para K-Means
        cv::Mat bestLabels;
        cv::Mat TextonDictionary;
        cv::Mat Kpoints;

        void reorient(cv::Mat &points, float angle, cv::Mat &reoriented_points);
        float gauss(float sigma, float mean, float value, int ord);
        void setFilter(cv::Mat &filter, cv::Mat &points, int scale, int phasex, int phasey);
        void setGaussianFilter(cv::Mat &filter, cv::Mat &points, float sigma);
        void setLoGFilter(cv::Mat &filter, cv::Mat &points, float sigma);
        //Leave samples to the filter size: crops them, assuming they are centered. If pair width or height they are scaled for fitting the filters size
        void prepareSamplesCrop(std::string dir_name);
        void prepareSamplesFromPoints(cv::Mat &image, std::deque<cv::Point2i> &points);
        void normalizeFilters();
        void calculateFilterResponses();
        void normalizeResponse(std::vector<float>& response);
        void applyKmeans(int size);

        static float applyFilter(cv::Mat &sample, cv::Mat &filter, int size);
        static void normalizeFilter(cv::Mat &filter, cv::Mat &norm_filter, int size);
        static void normalizeBGRtoGray(cv::Mat &in, cv::Mat &out, int size);
        static cv::Mat filterToShow(cv::Mat &filter);
};

#endif // FILTERBANK_H
