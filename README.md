# ViT_from_scratch
From scratch implementation of the paper: "[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)" ICLR 2021

### Features:
- ViT with manual patch encoding
- ViT with CNN patch encoding
- Train/validation loop for MNIST
- C++ deployment of the model with TorchScript
- Docker container with Python and C++ dependencies

Note: This project is for educational purposes and does not aim at performance.

### Run Training
```
cd python && python train.py
```

### Run Evaluation
```
cd python && python evaluate.py
```
Note: a pretrained model is available at _artifacts/model.pt_


### Run C++ Inference
Build and run the docker container
```
cd docker
bash build_docker.sh
bash run_docker.sh
```
Build and run the target code
```
cd application
bash build.sh
./build/main ../artifacts/model_ts.pt ../artifacts/img/single.png
```
