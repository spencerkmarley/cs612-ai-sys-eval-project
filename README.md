# <div align="center">CS612 AI Systems Evaluation Group 1: The Deep Fakers</div>
<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
<br>
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)
![Generic badge](https://img.shields.io/badge/STATUS-INPROGRESS-<COLOR>.svg)

</div>
<p align="center">
  <img src="data/deepfake.png" width="700">
</p>

Keeping deep learning models honest through testing.
#### <ins>__Members__</ins><br>
[Chng Kian Woon Gerry](mailto:gerry.chng.2020@mitb.smu.edu.sg)<br>
[He Chen](mailto:chen.he.2020@mitb.smu.edu.sg)<br>
[Lim Hsien Yong](mailto:hy.lim.2021@mitb.smu.edu.sg)<br>
[Spencer Keith Marley](mailto:skmarley.2021@mitb.smu.edu.sg)<br>

# Getting started!

https://user-images.githubusercontent.com/7207877/199639102-6544b5f1-dfbf-48db-a384-c5a9849f13fc.mp4

<br>
1. Git clone https://github.com/spencerkmarley/cs612-ai-sys-eval-project <br>
2. At your terminal, type "cd cs612-ai-sys-eval-project" <br>
3. Upload your model into models/subject <br>
4. Open utils/config.py and edit the parameters you want <br>
5. In terminal type python3 main.py <br>
 
## <div align="center">Project Directory Structure</div>
```
.
├── data
│   ├── MNIST/raw
│   ├── TriggerImg
│   ├── TriggerImg.zip
├── model tests         <- test functions to ascertain model backdoor/s
├── models              <- repository of model architectures
│   ├── benign   
│   ├── definitions
│   ├── subject
│   ├── train
├── notebooks           <- demonstration of model tests and functions
├── papers              <- papers referenced
├── trigger_clean       <- pt models
```
## <div align="center">Gallery of backdoors found</div>
#### <ins>__Backdoors from CIFAR10 subject model__</ins><br>
<p align="center">
  <img src="data/CIFAR10_backdoors.png" width="1000">
</p>
#### <ins>__Backdoors from our CIFAR10 subject model__</ins><br>
<p align="center">
  <img src="data/CIFAR10_Inv_triggers.png" width="1000">
</p>
<p align="center">
  <img src="data/Badnet_triggers.png" width="1000">
</p>
