#ifndef UTILS_H
#define UTILS_H
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
namespace my_utils
{
    cv::Mat load_image_from_file(const std::string &image_path);
    void save_image_to_file(cv::Mat img);
    cv::Mat resize_image(cv::Mat img, int new_width, int new_height);
    torch::Tensor to_tensor(const cv::Mat& img);

}
#endif // !UTILS_H