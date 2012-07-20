DSMGPUAcc
=========

software DSM, GPU Accelerator platform, CUDA enabled, regular data shape

DSMGPUAcc is a library used to provide a unified address space between CPU and
GPU (typically now is CUDA) to programmers. Now it support a model that multiple
threads running and each thread take care of a CUDA device.

We will investigate the new feature provided by NVidia that single thread take 
responsibility of multiple CUDA devices in the furture.
