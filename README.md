<p align="center">
<h1 align="center"><strong>F3D-Gaus: Feed-forward 3D-aware Generation on ImageNet with Cycle-Consistent Gaussian Splatting</strong></h1>
<!-- <h3 align="center">Arxiv 2025</h3> -->

<p align="center">
    <a href="https://w-ted.github.io/">Yuxin Wang</a><sup>1</sup>,</span>
    <a href="https://wuqianyi.top/">Qianyi Wu</a><sup>2</sup>,
    <a href="https://www.danxurgb.net/">Dan Xu</a><sup>1✉️</sup>
    <br>
        <sup>1</sup>Hong Kong University of Science and Technology,
        <sup>2</sup>Monash University
</p>


<div align="center">
    <a href='https://arxiv.org/abs/2501.06714' target="_blank"><img src='https://img.shields.io/badge/arXiv-2501.06714-b31b1b.svg'></a>  
    <a href='https://w-ted.github.io/publications/F3D-Gaus/' target="_blank"><img src='https://img.shields.io/badge/Project-Page-Green'></a>  
</div>
</p>

## Demo

<!-- ![demo-0](assets/output-0-clip.gif) -->
https://github.com/user-attachments/assets/db5c783c-1d1f-489e-8040-95353a4bb396


## Updates

- **`2025/01/12`**: We released this repo with the pre-trained model and inference code.

## Installation

```
git clone https://github.com/W-Ted/F3D-Gaus.git

cd F3D-Gaus
conda create -n f3d_gaus python=3.10.14 -y
conda activate f3d_gaus
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 # pytorch=2.0.1=py3.10_cuda11.7_cudnn8.5.0_0
pip install -r requirements.txt

# GOF
cd src/gaussian-splatting
pip install submodules/diff-gof-rasterization
pip install submodules/simple-knn/

# tetra-nerf for triangulation (mesh extraction)
cd submodules/tetra-triangulation
conda install cmake -y
conda install conda-forge::gmp
conda install conda-forge::cgal
cmake .
# you can specify your own cuda path
# export CPATH=/usr/local/cuda-11.3/targets/x86_64-linux/include:$CPATH
make 
pip install -e .
```

## Pre-trained model
We provide the pre-trained model [here(~720MB)](https://drive.google.com/file/d/1Uar3kyI5Oi5f3cZytUl5YKBkcg4HNALz/view?usp=sharing). You could download it to the ''pretrained_models'' directory. 

```
cd pretrained_models
pip install gdown && gdown 'https://drive.google.com/uc?id=1Uar3kyI5Oi5f3cZytUl5YKBkcg4HNALz'
cd ..
```


## Inference 
We provide two scripts for inference of F3D-Gaus: one for novel view synthesis and the other for subsequent mesh extraction. 
```
# single-image novel view synthesis
bash scripts/test_nvs.sh 

# single-image mesh extraction
bash scripts/test_mesh.sh
```

## Acknowledgements

This project is built upon [G3DR](https://preddy5.github.io/g3dr_website/) and [Splatter-Image](https://github.com/szymanowiczs/splatter-image). The 3DGS representation is borrowed from [GOF](https://niujinshuchong.github.io/gaussian-opacity-fields/). Kudos to these researchers. 

## Citation

```BibTeX
@article{wang2025f3dgaus,
    title={F3D-Gaus: Feed-forward 3D-aware Generation on ImageNet with Cycle-Consistent Gaussian Splatting},
    author={Wang, Yuxin and Wu, Qianyi and Xu, Dan},
    journal={arXiv preprint arXiv:},
    year={2025}
}
```
