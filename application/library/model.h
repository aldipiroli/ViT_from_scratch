#ifndef MODEL_H
#define MODEL_H
#include <torch/script.h>
#include <torch/torch.h>
#include <string>

struct ModelConfig
{
    int patch_size = 4;
};

struct ImageParameters
{
    int height = 28;
    int width = 28;
    int channels = 1;
};

class ViT
{
public:
    ViT(ModelConfig model_config);
    void load_model(const std::string model_path);
    void run_inference_on_image(const char *img_path);

private:
    ModelConfig model_config_;
    torch::jit::script::Module module_;
    ImageParameters img_parameters_;
};

#endif // !MODEL_H