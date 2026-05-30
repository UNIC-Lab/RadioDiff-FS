# RadioDiff-FS: Physics-Informed Manifold Alignment in Few-Shot Diffusion Models for High-Fidelity Radio Map Construction

---
## 📡 Welcome to the RadioDiff Family

> Radio map construction via generative diffusion models — UNIC Lab, Xidian University

---

### 🔷 Base Backbone

**RadioDiff** — *The foundational diffusion model for radio map construction.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/10764739) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff) &nbsp;|&nbsp; ![IEEE TCCN](https://img.shields.io/badge/IEEE-TCCN%202025-blue)

---

### 🔬 Physics-Informed Extensions

**RadioDiff-k²** — *PINN-enhanced diffusion guided by the Helmholtz equation.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11278649) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff-k) &nbsp;|&nbsp; ![IEEE JSAC](https://img.shields.io/badge/IEEE-JSAC%202026-blue)

**iRadioDiff** — *Indoor radio map construction with physical information integration.*
&nbsp;&nbsp;📄 [Paper](https://arxiv.org/abs/2511.20015) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/iRadioDiff) &nbsp;|&nbsp; ![IEEE ICC](https://img.shields.io/badge/IEEE-ICC%202026-blue) &nbsp;![Best Paper](https://img.shields.io/badge/🏆-Best%20Paper%20Award-orange)

---

### ⚡ Efficiency & Dynamics

**RadioDiff-Turbo** — *Efficiency-enhanced RadioDiff for accelerated inference.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/abstract/document/11152929/) &nbsp;|&nbsp; ![INFOCOM Workshop](https://img.shields.io/badge/IEEE-INFOCOM%20Wksp%202025-lightgrey)

**RadioDiff-Flux** — *Adaptive reconstruction under dynamic environments and base station location changes.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11282987/) &nbsp;|&nbsp; ![IEEE TCCN](https://img.shields.io/badge/IEEE-TCCN%202026-blue)

---

### 🌐 Extended Scenarios

**RadioDiff-3D** — *3D radio map construction with the UrbanRadio3D dataset.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11083758) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/UrbanRadio3D) &nbsp;|&nbsp; ![IEEE TNSE](https://img.shields.io/badge/IEEE-TNSE%202025-blue)

**RadioDiff-FS** — *Few-shot learning for radio map construction with limited measurements.*
&nbsp;&nbsp;📄 [Paper](https://arxiv.org/abs/2603.18865) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff-FS) &nbsp;|&nbsp; ![arXiv](https://img.shields.io/badge/arXiv-preprint-lightgrey)

---

### 📶 Sparse Measurement & Localization

**RadioDiff-Inverse** — *Sparse measurement-based radio map recovery for ISAC applications.*
&nbsp;&nbsp;📄 [Paper](https://arxiv.org/abs/2504.14298) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/radiodiff-inverse) &nbsp;|&nbsp; ![IEEE TWC](https://img.shields.io/badge/IEEE-TWC%202026-blue)

**RadioDiff-Loc** — *Sparse measurement-based NLoS localization using diffusion models.*
&nbsp;&nbsp;📄 [Paper](https://www.arxiv.org/abs/2509.01875) &nbsp;|&nbsp; ![arXiv](https://img.shields.io/badge/arXiv-preprint-lightgrey)

---

> 📚 For a comprehensive categorized overview of radio map research, visit [**Awesome-Radio-Map-Categorized**](https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized).


---

##  Before Starting

1. install torch
~~~
conda create -n radiodiff python=3.9
conda avtivate radiodiff
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
~~~
2. install other packages.
~~~
pip install -r requirement.txt
~~~
3. prepare accelerate config.
~~~
accelerate config # HOW MANY GPUs YOU WANG TO USE.
~~~

##  Prepare Data

##### We used the [RadioMapSeer](https://radiomapseer.github.io/) dataset for model training and testing.

- The data structure should look like:

```commandline
|-- $RadioMapSeer
|   |-- gain
|   |-- |-- carsDPM
|   |-- |-- |-- XXX_XX.PNG
|   |-- |-- |-- XXX_XX.PNG
|   ...
|   |-- png
|   |-- |-- buildings_complete
|   |-- |-- |-- XXX_XX.PNG
|   |-- |-- |-- XXX_XX.PNG
|	...
```
## :tada: Training
1. train the first stage model (AutoEncoder):
~~~
accelerate launch train_vae.py --cfg ./configs/first_radio.yaml
~~~
2. train latent diffusion-edge model:
~~~
accelerate launch train_cond_ldm.py --cfg ./configs/radio_train.yaml
~~~
3. fine-tune the pretrained model on IRT4 few-shot data:
~~~
accelerate launch train_cond_ldm_finetune.py --cfg ./configs/radio_train_irt4_finetune.yaml
~~~

## V. Inference.
~~~
python sample_cond_ldm.py --cfg ./configs/radio_test_finetune.yaml
~~~

