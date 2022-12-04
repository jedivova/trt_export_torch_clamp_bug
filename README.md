## Description

My model has few torch.clamp() func in it. exporting half-precision pytorch model (via path Pytorch-> ONNX -> TRT) then output of my model is zeroed.
I've created minimal repro code (please see attached notebook).

## Environment

**TensorRT Version**: 8.5.1.7
**NVIDIA GPU**: NVIDIA GeForce RTX 3080 Ti Laptop GPU
**NVIDIA Driver Version**: 515.65.01
**CUDA Version**: 11.8
**CUDNN Version**: 8.6.0.163-1+cuda11.8
**Operating System**: ubuntu 20.4
**Python Version (if applicable)**: 3.8.10
**PyTorch Version (if applicable)**: 1.12.1+cu116


## Relevant Files

<!-- Please include links to any models, data, files, or scripts necessary to reproduce your issue. (Github repo, Google Drive/Dropbox, etc.) -->


## Steps To Reproduce

just launch the code from jupyter notebook.
You can change "HALF=True" to False -> output will not be zeroed (as expected).

