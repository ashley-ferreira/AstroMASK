## AstroMASK: Astronomy Masked Autoencoder Self-supervised Knowledge

#### Applying Self-Supervised Representation Learning to the Ultraviolet Near Infrared Optical Northern Survey

**NOTE:** This repository is very much still a work in progress. I plan to upload my mid-project written report and I need to do one more big update to the code to reflect the software used to produce results in my written report. To do this I need to regain access to CANFAR and that should happen this week.

### About 
The Ultraviolet Near Infrared Optical Northern Survey (UNIONS) uses observations from three telescopes in Hawaii and aims to answer some of the most fundamental questions in astrophysics such as determining the properties of dark matter and dark energy, as well as the growth of structure in the Universe. However, being able to effectively search through and categorize the data in order to extract these insights can be cumbersome. This project hopes to exploit 
recent advances in a sub-field of Machine Learning (ML), called Self-Supervised Learning (SSL), including Masked Autoencoders (MAE) with Vision Transformer (ViT) 
backbones to train a model to produce meaningful lower-dimensional representations of astronomy observations without the need for explicit labels. These models have 
shown to be effective at performing similarity searches and take far fewer labels to fine-tune for downstream tasks such as strong lens detection. This report will cover the approach in more detail and touch on preliminary results of exploring these lower-dimensional representations.

### Presentations
[machine learning jamboree slides](https://docs.google.com/presentation/d/1yKvwjkmD0P0Yg99_3wGNmzTLRaY1v0lIrQrFEd7BDQQ/edit?usp=sharing) from November 2023

[PHYS 437A mid-term presentation](https://docs.google.com/presentation/d/1SYtv5tDFlD92CpKsPnbQ_WaoMLi82DHT/edit?usp=sharing&ouid=111358555577518196339&rtpof=true&sd=true) from October 2023

### Pre-requisites
- read access to the [dark3d repo](https://github.com/astroai/dark3d)
- a [Weights & Biases](https://docs.wandb.ai/quickstart#:~:text=Create%20an%20account%20and%20install,Python%203%20environment%20using%20pip%20.) account 
- CADC account
- read access to the following path on CADC's CANFAR
```
/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/10k_per_h5
```

### Quick Start
1. Follow this link to [CANFAR Science Portal](https://www.canfar.net/science-portal/) and log on to your CADC account
2. Launch a notebook with container image "skaha/astroml-notebook:latest"
3. Enter the following in their terminal with your CADC username in the place of YOUR_USERNAME
```
cadc-get-cert -u YOUR_USERNAME
```
4. Then run the following to launch a GPU session
```
curl -E .ssl/cadcproxy.pem 'https://ws-uv.canfar.net/skaha/v0/session?name=notebookgpu&cores=2&ram=16&gpus=1' -d image="images.canfar.net/skaha/astroml-gpu-notebook:latest"
```
5. Now, if you return to the [CANFAR Science Portal](https://www.canfar.net/science-portal/) you should see a new notebook session that has access to a GPU. If it stays greyed out this is likely because all GPUs are currently claimed.
6. Navigate to the directory you save you want to save code in and clone the following two repositories
```
git clone https://github.com/astroai/dark3d.git
git clone https://github.com/ashley-ferreira/mae.git
```
6. You are now ready to train the model! Navigate to the mae directory and run the script that trains the model
```
cd mae
python main_pretrain.py
```
which saves checkpoints in 
```
/arc/projects/unions/ssl/data/processed/unions-cutouts/ugriz_lsb/output_dir/DATETIME
```
where DATETIME is the time at which the code began running in UTC.
8. To analyze the outputs of this model, specifically the image reconstructions and the representations through UMAPs, t-SNEs, and similarity search, then run the notebooks found in mae/demo. 

### Acknowledgements
Many others have contributed to this effort including my supervisors Sebastien Fabbro and Mike Hudson, as well as Spencer Bialek, Nat Comeau, Nick Heesters, and Leonardo Ferreira. 

This research used the facilities of the CADC operated by the National Research Council of Canada with the support of the Canadian Space Agency. Without the CADC's CANFAR platform, none of this work would have been possible, the platform was used to host and access the data as well as perform all computational work needed. 

All data used for this project is from UNIONS and so this survey has been instrumental in every aspect of this project.

### Built With

[![Python][python]][python-url]
[![Notebook][notebook]][notebook-url] 
[![PyTorch][pytorch]][pytorch-url]
[![WandB][wandb]][wandb-url] 

W\&B experiment tracking software gives free student accounts and that was tremendously helpful to be able to use to debug and keep track of different experiments.

Finally, this work project has heavily relied on open-source software. All the programming was done in Python and made use of its many associated packages including NumPy, Matplotlib, and two key contributions from Meta AI: PyTorch and Faiss. Additionally, this work made use of Astropy a community-developed core Python package and an ecosystem of tools and resources for astronomy.

This code is also forked from another repository for which information is available below:

## Masked Autoencoders: A PyTorch Implementation

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


This is a PyTorch/GPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```

* The original implementation was in TensorFlow+TPU. This re-implementation is in PyTorch+GPU.

* This repo is a modification on the [DeiT repo](https://github.com/facebookresearch/deit). Installation and preparation follow that repo.

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

### Catalog

- [x] Visualization demo
- [x] Pre-trained checkpoints + fine-tuning code
- [x] Pre-training code

### Visualization demo

Run our interactive visualization demo using [Colab notebook](https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb) (no GPU needed):
<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/147859292-77341c70-2ed8-4703-b153-f505dcb6f2f8.png" width="600">
</p>

### Fine-tuning with pre-trained checkpoints

The following table provides the pre-trained checkpoints used in the paper, converted from TF/TPU to PT/GPU:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">ViT-Huge</th>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth">download</a></td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>8cad7c</tt></td>
<td align="center"><tt>b8b06e</tt></td>
<td align="center"><tt>9bdbb0</tt></td>
</tr>
</tbody></table>

The fine-tuning instruction is in [FINETUNE.md](FINETUNE.md).

By fine-tuning these pre-trained models, we rank #1 in these classification tasks (detailed in the paper):
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-B</th>
<th valign="bottom">ViT-L</th>
<th valign="bottom">ViT-H</th>
<th valign="bottom">ViT-H<sub>448</sub></th>
<td valign="bottom" style="color:#C0C0C0">prev best</td>
<!-- TABLE BODY -->
<tr><td align="left">ImageNet-1K (no external data)</td>
<td align="center">83.6</td>
<td align="center">85.9</td>
<td align="center">86.9</td>
<td align="center"><b>87.8</b></td>
<td align="center" style="color:#C0C0C0">87.1</td>
</tr>
<td colspan="5"><font size="1"><em>following are evaluation of the same model weights (fine-tuned in original ImageNet-1K):</em></font></td>
<tr>
</tr>
<tr><td align="left">ImageNet-Corruption (error rate) </td>
<td align="center">51.7</td>
<td align="center">41.8</td>
<td align="center"><b>33.8</b></td>
<td align="center">36.8</td>
<td align="center" style="color:#C0C0C0">42.5</td>
</tr>
<tr><td align="left">ImageNet-Adversarial</td>
<td align="center">35.9</td>
<td align="center">57.1</td>
<td align="center">68.2</td>
<td align="center"><b>76.7</b></td>
<td align="center" style="color:#C0C0C0">35.8</td>
</tr>
<tr><td align="left">ImageNet-Rendition</td>
<td align="center">48.3</td>
<td align="center">59.9</td>
<td align="center">64.4</td>
<td align="center"><b>66.5</b></td>
<td align="center" style="color:#C0C0C0">48.7</td>
</tr>
<tr><td align="left">ImageNet-Sketch</td>
<td align="center">34.5</td>
<td align="center">45.3</td>
<td align="center">49.6</td>
<td align="center"><b>50.9</b></td>
<td align="center" style="color:#C0C0C0">36.0</td>
</tr>
<td colspan="5"><font size="1"><em>following are transfer learning by fine-tuning the pre-trained MAE on the target dataset:</em></font></td>
</tr>
<tr><td align="left">iNaturalists 2017</td>
<td align="center">70.5</td>
<td align="center">75.7</td>
<td align="center">79.3</td>
<td align="center"><b>83.4</b></td>
<td align="center" style="color:#C0C0C0">75.4</td>
</tr>
<tr><td align="left">iNaturalists 2018</td>
<td align="center">75.4</td>
<td align="center">80.1</td>
<td align="center">83.0</td>
<td align="center"><b>86.8</b></td>
<td align="center" style="color:#C0C0C0">81.2</td>
</tr>
<tr><td align="left">iNaturalists 2019</td>
<td align="center">80.5</td>
<td align="center">83.4</td>
<td align="center">85.7</td>
<td align="center"><b>88.3</b></td>
<td align="center" style="color:#C0C0C0">84.1</td>
</tr>
<tr><td align="left">Places205</td>
<td align="center">63.9</td>
<td align="center">65.8</td>
<td align="center">65.9</td>
<td align="center"><b>66.8</b></td>
<td align="center" style="color:#C0C0C0">66.0</td>
</tr>
<tr><td align="left">Places365</td>
<td align="center">57.9</td>
<td align="center">59.4</td>
<td align="center">59.8</td>
<td align="center"><b>60.3</b></td>
<td align="center" style="color:#C0C0C0">58.0</td>
</tr>
</tbody></table>

### Pre-training

The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/

[notebook]: https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter
[notebook-url]: https://jupyter.org/

[wandb]: https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white
[wandb-url]: https://wandb.ai/site

[pytorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[pytorch-url]: https://pytorch.org/
