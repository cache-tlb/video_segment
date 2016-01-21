#ifndef OPENCV_API_H
#define OPENCV_API_H

#include <opencv2/opencv.hpp>
#include <vector>

bool GraphBasedHierarchicalSegmentation(const std::vector<cv::Mat> &frames, float c, float c_reg, int min_size, float sigma, int hie_num, int output_level, std::vector<cv::Mat> &output_frame_labels);

#endif