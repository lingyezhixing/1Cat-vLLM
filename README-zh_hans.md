# **1Cat-vLLM-0.0.1——一个 vLLM SM70 AWQ 分支**

一个针对 **Tesla V100 / SM70** 的实验性 vLLM 分支，支持在 **Volta 架构 GPU 上进行 AWQ 4-bit 推理**。

上游 vLLM 的默认路径中，AWQ 内核要求 **SM75+** 架构。本分支集成了 **lmdeploy TurboMind SM70 WMMA 内核** 以及一系列针对 SM70 的运行时修复，使得 V100 能够提供对现代 AWQ 模型（包括稠密模型、MoE 模型及部分多模态模型）的推理服务。

## **本分支新增功能**

* 为 **SM70 / Tesla V100** 提供 AWQ 4-bit 支持  
* 在 V100 上实现稠密 (Dense) 和 MoE AWQ 的执行路径  
* 在部分压缩张量 (compressed-tensors) MoE 路径中复用 SM70 AWQ 内核  
* 针对 SM70 的 Triton 注意力机制以及 MLA/GDN 运行时修复  
* 兼容 torch.compile 和 CUDA graphs  
* 通过标准的 vLLM 入口提供兼容 OpenAI 的 API 服务

本仓库基于 vLLM，并在尽可能保留原始项目结构和服务 API 的基础上进行开发。

## **基准测试 / 性能数据**

如需查看简洁的基准测试数据（解码/预填充对比及输入长度扩展曲线），请参阅：

* [effort/README.md](https://www.google.com/search?q=effort/README.md)

预览（画廊）：

| 总览对比 | Qwen3-coder-next-80b（1 卡） | Qwen3.5-122B-A10B-AWQ（4 卡） |
| :---- | :---- | :---- |
|  |  |  |

## **已验证的环境栈**

本 README 中的命令已在以下环境中完成验证：

* 操作系统: Ubuntu 24.04.4 LTS  
* Python 版本: 3.12.13  
* CUDA Toolkit: 12.8  
* PyTorch: 2.9.1+cu128  
* Triton: 3.5.1  
* 驱动版本: 570.211.01  
* 显卡: 4 x Tesla V100-SXM2-16GB

## **核心思路**

本分支主要修改了以下两个方面：

1. **SM70 上的 AWQ**  
   * 集成 TurboMind SM70 原生 8x8x4 半精度 MMA 内核。  
   * 扩展 vLLM AWQ 量化层，使 V100 能够运行通常需要 Turing 或更新架构才能运行的 AWQ 模型。  
2. **SM70 运行时修复**  
   * 针对 Volta 架构的 Triton 注意力机制调优  
   * MLA/GDN 兼容性工作  
   * 针对此类硬件的 CUDA graph 和 torch.compile 调整

## **当前重点**

本分支主要针对：

* Tesla V100 / SM70 用户  
* AWQ 4-bit 部署  
* Qwen3 / Qwen3.5 风格模型  
* 小显存多卡服务架构，例如 4 x 16GB 环境

## **重要的运行时注意事项**

* **首个实际请求的速度不能代表稳定状态下的速度**。在 V100 上，第一个请求花费 **1 到 3 分钟** 来编译内核、构建计算图和预热执行路径是非常正常的。  
* 在已验证的 4 x V100 16GB 机器上，最稳定的运行时设置是：  
  * VLLM\_DISABLE\_PYNCCL=1  
  * \--disable-custom-all-reduce  
  * \--gpu-memory-utilization 0.80  
* 对于在 4 x 16GB 环境运行的 Qwen3.5-35B-A3B-AWQ 模型，设置 gpu\_memory\_utilization=0.92 可以启动服务，但在处理第一个实际请求时并不稳定。  
* 在小显存显卡上进行**纯文本服务**时，以下设置有助于提升性能：  
  * \--skip-mm-profiling  
  * \--limit-mm-per-prompt '{"image":0,"video":0}'

## **快速开始**

### **1\. 安装 CUDA 12.8**

在 Ubuntu 24.04 上使用 NVIDIA 官方仓库：

wget \[https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86\_64/cuda-keyring\_1.1-1\_all.deb\](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86\_64/cuda-keyring\_1.1-1\_all.deb)  
sudo dpkg \-i cuda-keyring\_1.1-1\_all.deb  
sudo apt update  
sudo apt install \-y cuda-toolkit-12-8

如果机器上同时安装了 CUDA 13.x，请在构建时强制将 CUDA 路径指向 12.8：

export CUDA\_HOME=/usr/local/cuda-12.8  
export PATH=$CUDA\_HOME/bin:$PATH  
export LD\_LIBRARY\_PATH=$CUDA\_HOME/lib64:${LD\_LIBRARY\_PATH:-}  
hash \-r  
nvcc \-V

### **2\. 创建 conda 环境**

source /path/to/miniconda3/etc/profile.d/conda.sh  
conda create \-y \-n vllm-py312 python=3.12  
conda activate vllm-py312  
python \-m pip install \--upgrade pip setuptools wheel

### **3\. 准备 lmdeploy 源码依赖**

本分支目前依赖 lmdeploy 中特定的 TurboMind 源码来构建 SM70 AWQ 内核。

在构建之前，请初始化指定的 lmdeploy 子模块：

cd /path/to/vllm

git submodule update \--init \--recursive lmdeploy

注意：

* 该子模块指向 1CatAI/lmdeploy 分支，其中已包含了本分支所需的 SM70 兼容性修改。  
* 考虑到构建的可重复性，它被锁定在指定的 commit 159b0ab8b7b7442082bcab8b8a7dd008c40c7b58。  
* 如果您在克隆仓库时没有携带子模块，请务必在构建前运行上述命令。

### **4\. 安装依赖**

本仓库使用 requirements/ 目录分类管理依赖，而非单个 requirements.txt。

cd /path/to/vllm

python \-m pip install \-r requirements/build.txt  
python \-m pip install \-r requirements/cuda.txt  
python \-m pip install \-r requirements/common.txt

### **5\. 从源码构建**

经验证的 Release 风格构建使用了 **24 个 CMake 编译任务 (jobs)**。

cd /path/to/vllm  
source /path/to/miniconda3/etc/profile.d/conda.sh  
conda activate vllm-py312

export CUDA\_HOME=/usr/local/cuda-12.8  
export PATH=$CUDA\_HOME/bin:$PATH  
export LD\_LIBRARY\_PATH=$CUDA\_HOME/lib64:${LD\_LIBRARY\_PATH:-}  
export TORCH\_CUDA\_ARCH\_LIST="7.0"  
export MAX\_JOBS=24  
export CMAKE\_BUILD\_PARALLEL\_LEVEL=24  
export NVCC\_THREADS=1

python \-m pip install \-e . \--no-build-isolation

启用 CUDA 构建时，setup.py 会将 MAX\_JOBS 除以 NVCC\_THREADS。如果你想要完整的 cmake \--build \-j24 效果，请保持 MAX\_JOBS=24，并将 NVCC\_THREADS=1（或不设置该变量）。

### **6\. 验证环境**

python \- \<\<'PY'  
import torch, triton, vllm, sys  
print("python", sys.version.split()\[0\])  
print("torch", torch.\_\_version\_\_)  
print("torch\_cuda", torch.version.cuda)  
print("triton", triton.\_\_version\_\_)  
print("vllm", vllm.\_\_version\_\_)  
PY

### **7\. 构建预编译 wheel 包**

如果你想要一个可分发的预编译包，而不是可编辑的源码安装，可以使用同样的 24-job 设置来构建 wheel 包：

cd /path/to/vllm  
source /path/to/miniconda3/etc/profile.d/conda.sh  
conda activate vllm-py312

export CUDA\_HOME=/usr/local/cuda-12.8  
export PATH=$CUDA\_HOME/bin:$PATH  
export LD\_LIBRARY\_PATH=$CUDA\_HOME/lib64:${LD\_LIBRARY\_PATH:-}  
export TORCH\_CUDA\_ARCH\_LIST="7.0"  
export MAX\_JOBS=24  
export CMAKE\_BUILD\_PARALLEL\_LEVEL=24  
export NVCC\_THREADS=1

python \-m pip wheel . \--no-build-isolation \-w dist

此命令会将主程序 vllm-\*.whl 及其依赖的 wheel 包全部写入 dist/ 目录下，该 dist/ 目录可作为另一台机器上的本地 wheel 仓库 (wheelhouse)。

在另一台机器上从该本地目录安装 wheel 包：

source /path/to/miniconda3/etc/profile.d/conda.sh  
conda create \-y \-n vllm-py312 python=3.12  
conda activate vllm-py312

python \-m pip install \--no-index \\  
  \--find-links /path/to/vllm/dist \\  
  /path/to/vllm/dist/vllm-\*.whl

由于 wheel 包已经包含了编译好的扩展，因此通过 wheel 安装运行时，不再需要 lmdeploy 的源码树。

## **针对 V100 16GB 的推荐服务设置**

除非你明确需要其他配置，否则请使用此基准配置：

export VLLM\_DISABLE\_PYNCCL=1

推荐的启动参数：

\--attention-backend TRITON\_ATTN  
\--disable-custom-all-reduce  
\--compilation-config '{"cudagraph\_mode":"full\_and\_piecewise","cudagraph\_capture\_sizes":\[1\]}'  
\--gpu-memory-utilization 0.80  
\--max-model-len 512  
\--max-num-seqs 1  
\--max-num-batched-tokens 128

对于小显卡上的**纯文本**模式，建议补充添加：

\--skip-mm-profiling  
\--limit-mm-per-prompt '{"image":0,"video":0}'

## **启动示例**

### **Qwen3.5-27B-AWQ，TP2，纯文本模式**

CUDA\_VISIBLE\_DEVICES=0,1 \\  
VLLM\_DISABLE\_PYNCCL=1 \\  
python \-m vllm.entrypoints.openai.api\_server \\  
  \--model /path/to/Qwen3.5-27B-AWQ \\  
  \--served-model-name qwen35-27b \\  
  \--tensor-parallel-size 2 \\  
  \--dtype float16 \\  
  \--gpu-memory-utilization 0.80 \\  
  \--max-model-len 512 \\  
  \--max-num-seqs 1 \\  
  \--max-num-batched-tokens 128 \\  
  \--trust-remote-code \\  
  \--attention-backend TRITON\_ATTN \\  
  \--disable-custom-all-reduce \\  
  \--skip-mm-profiling \\  
  \--limit-mm-per-prompt '{"image":0,"video":0}' \\  
  \--compilation-config '{"cudagraph\_mode":"full\_and\_piecewise","cudagraph\_capture\_sizes":\[1\]}' \\  
  \--host 127.0.0.1 \\  
  \--port 8201

### **Qwen3.5-27B-AWQ，TP4，纯文本模式**

CUDA\_VISIBLE\_DEVICES=0,1,2,3 \\  
VLLM\_DISABLE\_PYNCCL=1 \\  
python \-m vllm.entrypoints.openai.api\_server \\  
  \--model /path/to/Qwen3.5-27B-AWQ \\  
  \--served-model-name qwen35-27b \\  
  \--tensor-parallel-size 4 \\  
  \--dtype float16 \\  
  \--gpu-memory-utilization 0.80 \\  
  \--max-model-len 512 \\  
  \--max-num-seqs 1 \\  
  \--max-num-batched-tokens 128 \\  
  \--trust-remote-code \\  
  \--attention-backend TRITON\_ATTN \\  
  \--disable-custom-all-reduce \\  
  \--skip-mm-profiling \\  
  \--limit-mm-per-prompt '{"image":0,"video":0}' \\  
  \--compilation-config '{"cudagraph\_mode":"full\_and\_piecewise","cudagraph\_capture\_sizes":\[1\]}' \\  
  \--host 127.0.0.1 \\  
  \--port 8200

### **Qwen3.5-35B-A3B-AWQ，TP4，纯文本模式**

CUDA\_VISIBLE\_DEVICES=0,1,2,3 \\  
VLLM\_DISABLE\_PYNCCL=1 \\  
python \-m vllm.entrypoints.openai.api\_server \\  
  \--model /path/to/Qwen3.5-35B-A3B-AWQ \\  
  \--served-model-name qwen35-35b \\  
  \--tensor-parallel-size 4 \\  
  \--dtype float16 \\  
  \--gpu-memory-utilization 0.80 \\  
  \--max-model-len 512 \\  
  \--max-num-seqs 1 \\  
  \--max-num-batched-tokens 128 \\  
  \--trust-remote-code \\  
  \--attention-backend TRITON\_ATTN \\  
  \--disable-custom-all-reduce \\  
  \--skip-mm-profiling \\  
  \--limit-mm-per-prompt '{"image":0,"video":0}' \\  
  \--compilation-config '{"cudagraph\_mode":"full\_and\_piecewise","cudagraph\_capture\_sizes":\[1\]}' \\  
  \--host 127.0.0.1 \\  
  \--port 8200

### **Qwen3.5-27B-AWQ，TP4，启用视觉功能**

此模型包含 vision\_config。要启用图像理解功能，请勿以纯文本模式启动服务。

CUDA\_VISIBLE\_DEVICES=0,1,2,3 \\  
VLLM\_DISABLE\_PYNCCL=1 \\  
python \-m vllm.entrypoints.openai.api\_server \\  
  \--model /path/to/Qwen3.5-27B-AWQ \\  
  \--served-model-name qwen35-27b \\  
  \--tensor-parallel-size 4 \\  
  \--dtype float16 \\  
  \--gpu-memory-utilization 0.80 \\  
  \--max-model-len 4096 \\  
  \--max-num-seqs 1 \\  
  \--max-num-batched-tokens 512 \\  
  \--trust-remote-code \\  
  \--attention-backend TRITON\_ATTN \\  
  \--disable-custom-all-reduce \\  
  \--limit-mm-per-prompt '{"image":1,"video":0}' \\  
  \--allowed-local-media-path /path/to/media \\  
  \--compilation-config '{"cudagraph\_mode":"full\_and\_piecewise","cudagraph\_capture\_sizes":\[1\]}' \\  
  \--host 127.0.0.1 \\  
  \--port 8000

注意事项：

* **请勿**在启用视觉功能的设置中传入 \--skip-mm-profiling 参数。  
* **请勿**设置 \--limit-mm-per-prompt '{"image":0,"video":0}'。  
* 在 4 x V100 16GB 的机器上，在尝试更大的上下文长度之前，建议保守起见先从 4096 / 512（最大模型长度 / 最大批处理 token 数）开始测试。

## **兼容 OpenAI 的请求格式**

### **纯文本请求**

curl \[http://127.0.0.1:8200/v1/chat/completions\](http://127.0.0.1:8200/v1/chat/completions) \\  
  \-H 'Content-Type: application/json' \\  
  \-d '{  
    "model": "qwen35-27b",  
    "messages": \[{"role":"user","content":"你好，简单介绍一下你自己。"}\],  
    "max\_tokens": 16,  
    "temperature": 0,  
    "stream": false  
  }'

### **图像请求**

远程图片：

curl \[http://127.0.0.1:8000/v1/chat/completions\](http://127.0.0.1:8000/v1/chat/completions) \\  
  \-H 'Content-Type: application/json' \\  
  \-d '{  
    "model": "qwen35-27b",  
    "messages": \[  
      {  
        "role": "user",  
        "content": \[  
          {"type": "text", "text": "Describe this image."},  
          {"type": "image\_url", "image\_url": {"url": "\[https://example.com/test.jpg\](https://example.com/test.jpg)"}}  
        \]  
      }  
    \],  
    "max\_tokens": 256,  
    "temperature": 0  
  }'

本地图片：

curl \[http://127.0.0.1:8000/v1/chat/completions\](http://127.0.0.1:8000/v1/chat/completions) \\  
  \-H 'Content-Type: application/json' \\  
  \-d '{  
    "model": "qwen35-27b",  
    "messages": \[  
      {  
        "role": "user",  
        "content": \[  
          {"type": "text", "text": "Describe this image."},  
          {"type": "image\_url", "image\_url": {"url": "file:///path/to/test.jpg"}}  
        \]  
      }  
    \],  
    "max\_tokens": 256,  
    "temperature": 0  
  }'

## **基准测试方法**

以下经过验证的吞吐量数据使用了：

* 兼容 OpenAI 格式的 /v1/chat/completions 接口  
* 流式响应 (streaming responses)  
* 1 次预热请求  
* 3 次测试请求  
* 提示词 (prompt) 长度约 128 个 token  
* 输出长度 32 个 token

重申：**在评估吞吐量时，请忽略第一次预热请求。**

## **4 x V100 16GB 热运行结果**

| 模型 | TP | 平均 TTFT | 平均 Decode TPS | 平均 Total TPS |
| :---- | :---- | :---- | :---- | :---- |
| Qwen3.5-27B-AWQ | 2 | 169.67 ms | 50.73 | 40.99 |
| Qwen3.5-27B-AWQ | 4 | 197.93 ms | 65.76 | 47.82 |
| Qwen3.5-35B-A3B-AWQ | 4 | 216.92 ms | 81.19 | 53.45 |

## **已知限制**

* 本分支专为 **SM70 / Tesla V100** 优化，并不针对所有硬件。  
* 上述结果是在保守设置下的 **4 x 16GB** 显卡上获得的。  
* 较长的上下文长度和多模态功能会迅速增加 16GB GPU 的显存压力。  
* Qwen3.5-35B-A3B-AWQ 在 4 x V100 16GB 环境中运行，设置 gpu\_memory\_utilization=0.92 时不稳定。  
* 如果您希望获得最大的上下文长度，请首先确认您的显存预算是否足以支撑当前模型和多模态功能的混合需求。

## **仓库说明**

* 最初的上游项目是 **vLLM**。  
* 本分支专注于 **SM70 AWQ 支持以及面向 V100 的运行时调优**。  
* 开发笔记和特定机器的复现细节保存在：  
  * [RUN\_V100\_CUDA128.md](https://www.google.com/search?q=RUN_V100_CUDA128.md)  
  * [CLAUDE.md](http://docs.google.com/CLAUDE.md)

## **致谢**

* [vLLM](https://github.com/vllm-project/vllm)  
* [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)

## **许可证**

本仓库遵循上游 vLLM 的许可证模式。详情请参阅 [LICENSE](https://www.google.com/search?q=LICENSE)。