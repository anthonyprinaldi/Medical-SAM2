<h1 align="center">● Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2</h1>

<p align="center">
    <a href="https://discord.gg/DN4rvk95CC">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <img src="https://img.shields.io/static/v1?label=license&message=GPL&color=white&style=flat" alt="License"/>
</p>

Medical SAM 2, or say MedSAM-2, is an advanced segmentation model that utilizes the [SAM 2](https://github.com/facebookresearch/segment-anything-2) framework to address both 2D and 3D medical
image segmentation tasks. This method is elaborated on the paper [Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2](https://arxiv.org/abs/2408.00874).

## 🔥 A Quick Overview 
 <div align="center"><img width="880" height="350" src="https://github.com/MedicineToken/Medical-SAM2/blob/main/vis/framework.png"></div>
 
## 🩻 3D Abdomen Segmentation Visualisation
 <div align="center"><img width="420" height="420" src="https://github.com/MedicineToken/Medical-SAM2/blob/main/vis/example.gif"></div>

## 🧐 Requirement

 Install the environment:

 ``pip install -e ".[dev]"``

 You can download SAM2 checkpoint from checkpoints folder:
 
 ``bash download_ckpts.sh``

 Also download the pretrain weights [here.](https://huggingface.co/jiayuanz3/MedSAM2_pretrain/tree/main)


 Further Note: We tested on the following system environment and you may have to handle some issue due to system difference.
```
Operating System: Rocky Linux 8.10
Pip Version: 24.2
Python Version: 3.11.5
```