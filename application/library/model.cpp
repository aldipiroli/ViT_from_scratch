#include "model.h"
#include "utils.h"
#include <iostream>
#include <opencv2/opencv.hpp>

ViT::ViT(ModelConfig model_config)
{
    model_config_ = model_config;
}

void ViT::load_model(const std::string model_path)
{
    module_ = torch::jit::load(model_path);
    std::cout << "Model loaded successfully.\n";
}

void ViT::run_inference_on_image(const char *img_path)
{
    // Load Image
    cv::Mat full_img;
    full_img = my_utils::load_image_from_file(img_path);
    cv::Mat img = my_utils::resize_image(full_img, img_parameters_.height, img_parameters_.width);
    my_utils::save_image_to_file(img);

    // Convert to tensor
    torch::Tensor img_tensor = my_utils::to_tensor(img);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor = img_tensor.unsqueeze(0); // BxCxHxW

    // Run Inference
    torch::NoGradGuard no_grad;
    module_.eval();
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_tensor);
    at::Tensor output = module_.forward(inputs).toTensor();

    // Get class and probability
    namespace F = torch::nn::functional;
    at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
    std::tuple<at::Tensor, at::Tensor> top_tensor = output_sm.topk(1);
    int pred_class = std::get<1>(top_tensor).item<int>();
    float pred_prob = std::get<0>(top_tensor).item<float>();
    std::cout << "Class: " << pred_class << ", prob: " << pred_prob << std::endl;
}
