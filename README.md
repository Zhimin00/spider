
## Table of Contents

- [Table of Contents](#table-of-contents)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Checkpoints](#checkpoints)
  - [Demo / How to use](#demo--how-to-use)
  - [Reproducing results](#reproducing-results)
- [License](#license)




## Get Started

### Installation

1. Clone SPIDER.
```bash
git clone --recursive https://github.com/Zhimin00/spider.git
cd spider
# if you have already cloned spider:
# git submodule update --init --recursive
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n spider python=3.11 cmake=3.14.0
conda activate spider
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add required packages for visloc.py
pip install -r dust3r/requirements_optional.txt
# conda install -c conda-forge glib
```

3. compile and install ASMK
```bash
pip install cython

git clone https://github.com/jenicek/asmk
cd asmk/cython/
cythonize *.pyx
cd ..
pip install .  # or python3 setup.py build_ext --inplace
cd ..
```

4. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

### Checkpoints
Download checkpoints from [One Drive](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zshao14_jh_edu/IQCjOrmorGYmT4EV_0VZVZkuAVBqvec_a7f7KbAHx7rjUXw?e=oKBe2I)

### Demo / How to use

We provide some demos and visualization codes in [demo_spiderkpts.py](demo_spiderkpts.py)

### Reproducing results

```
python eval_relpose.py --coarse_size 512 --fine_size 1600
```

## License

The code is distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.
