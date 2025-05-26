#include <iostream>
#include "library/model.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path>\n";
        return 1;
    }

    ModelConfig config;
    ViT model(config);

    model.load_model(argv[1]);
    model.run_inference_on_image(argv[2]);

    return 0;
}