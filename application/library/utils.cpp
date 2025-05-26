#include "utils.h"
namespace my_utils
{
    cv::Mat load_image_from_file(const std::string &image_path)
    {
        cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE); // Load as grayscale
        if (img.empty())
        {
            std::cerr << "Error loading image: " << image_path << std::endl;
            return img;
        }
        std::cout << "Loaded image!\n";
        return img;
    }

    void save_image_to_file(cv::Mat img)
    {
        std::string filename = "tmp.png";
        cv::imwrite(filename, img);
        std::cout << "Image saved as " << filename << std::endl;
    }

    cv::Mat resize_image(cv::Mat img, int new_width, int new_height)
    {
        cv::resize(img, img, cv::Size(new_width, new_height), 0, 0, cv::INTER_AREA);
        return img;
    }

    torch::Tensor to_tensor(const cv::Mat &img)
    {
        cv::Mat img_float;
        img.convertTo(img_float, CV_32F, 1.0 / 255);
        auto tensor = torch::from_blob(
            img_float.data,
            {img.rows, img.cols, img.channels()},
            torch::kFloat32);
        return tensor.clone();
    }

}