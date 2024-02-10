# devcontainer 说明

两个方案：

1. pytorch_docker + pip
2. python + poetry

## pytorch_docker + pip

+ [pytorch_docker](https://github.com/cnstark/pytorch-docker) 生成纯洁的 pytorch+cuda+python 的 docker 镜像
+ vscode devcontainer 生成最终的 vscode 开发环境
+ 使用 pip 进行包管理
  
### Generate Build Script

Generate build script by following command (available versions see [Available Versions](#Available-Versions)):

```shell
python generate_build_script.py --os <ubuntu or centos> --os-version <e.g. 20.04, 8> --python <e.g. 3.9.12> --pytorch <e.g. 1.9.1> --cuda <e.g. 11.1, cpu>
```

```shell
usage: generate_build_script.py [-h] --os OS --os-version OS_VERSION --python PYTHON --pytorch PYTORCH [--cuda CUDA] [--cuda-flavor CUDA_FLAVOR]

Generate docker build script.

optional arguments:
  -h, --help            show this help message and exit
  --os OS               Operating system.
  --os-version OS_VERSION
                        Operating system version.
  --python PYTHON       Python version.
  --pytorch PYTORCH     Pytorch version.
  --cuda CUDA           CUDA version, `cpu` means CPU version.
  --cuda-flavor CUDA_FLAVOR
                        CUDA flavor, `runtime` or `devel`, default is None, means use base image
```

### Build Pytorch Docker Image

```
scripts/build_xxx.sh
```

## python + poetry

+ 使用官方的 python 镜像编写 Dockerfile
+ 容器中，使用 poetry 进行包管理

优势：包的管理更加清晰，具体信息储存在 pyproject.toml 与 poetry.lock 中。

## vscode .devcontainer

include:
+ vscode extension
+ vscode settings
+ vscode devcontainer settings
+ requirements.txt