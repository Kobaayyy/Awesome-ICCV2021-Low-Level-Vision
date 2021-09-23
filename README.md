# Awesome-ICCV2021-Low-Level-Vision[![Awesome](https://camo.githubusercontent.com/13c4e50d88df7178ae1882a203ed57b641674f94/68747470733a2f2f63646e2e7261776769742e636f6d2f73696e647265736f726875732f617765736f6d652f643733303566333864323966656437386661383536353265336136336531353464643865383832392f6d656469612f62616467652e737667)](https://github.com/sindresorhus/awesome)
A Collection of Papers and Codes for ICCV2021 Low Level Vision and Image Generation

整理汇总下2021年ICCV中图像生成（Image Generation）和底层视觉(Low-Level Vision)任务相关的论文和代码，包括图像生成，图像编辑，图像风格迁移，图像翻译，图像修复，图像超分及其他底层视觉任务。大家如果觉得有帮助，欢迎star~~

**参考或转载请注明出处**

ICCV2021官网：[https://iccv2021.thecvf.com/](https://iccv2021.thecvf.com/)

开会时间：2021年10月11日-10月17日


**【Contents】**
- [1.图像生成（Image Generation）](#1.图像生成)
- [2.图像编辑（Image Manipulation/Image Editing）](#2.图像编辑)
- [3.图像风格迁移（Image Transfer）](#3.图像风格迁移)
- [4.图像翻译（Image to Image Translation）](#4.图像翻译)
- [5.图像修复（Image Inpaiting/Image Completion）](#5.图像修复)
- [6.图像超分辨率（Image Super-Resolution）](#6.图像超分辨率)
- [7.其他底层视觉任务（Other Low Level Vision）](#7.其他底层视觉任务)


<a name="1.图像生成"></a>
# 1.图像生成（Image Generation）
## Multiple Heads are Better than One: Few-shot Font Generation with Multiple Localized Experts
- Paper：https://arxiv.org/abs/2104.00887
- Code：https://github.com/clovaai/mxfont
- 小样本字体生成
## PIRenderer: Controllable Portrait Image Generation via Semantic Neural Rendering
- Code：https://github.com/RenYurui/PIRender
## Toward Spatially Unbiased Generative Models
- Code：https://github.com/jychoi118/toward_spatial_unbiased
## Disentangled Lifespan Face Synthesis
- Paper：https://arxiv.org/abs/2108.02874
- Code：https://github.com/clovaai/mxfont
## Handwriting Transformers
- Paper：https://arxiv.org/abs/2104.03964
- Code：https://github.com/ankanbhunia/Handwriting-Transformers
## Diagonal Attention and Style-based GAN for Content-Style Disentanglement in Image Generation and Translation
- Paper：https://arxiv.org/abs/2103.16146
## ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinement
- Paper：https://arxiv.org/abs/2104.02699
- Code：https://github.com/yuval-alaluf/restyle-encoder
## Paint Transformer: Feed Forward Neural Painting with Stroke Prediction
- Paper：https://arxiv.org/abs/2108.03798
- Code：https://github.com/huage001/painttransformer
## GAN Inversion for Out-of-Range Images with Geometric Transformations
- Paper：https://arxiv.org/abs/2108.08998

<a name="2.图像编辑"></a>
# 2.图像编辑（Image Manipulation/Image Editing）
## EigenGAN: Layer-Wise Eigen-Learning for GANs
- Paper：https://arxiv.org/abs/2104.12476
- Code：https://github.com/LynnHo/EigenGAN-Tensorflow
## From Continuity to Editability: Inverting GANs with Consecutive Images
- Paper：https://arxiv.org/abs/2107.13812
- Code：https://github.com/cnnlstm/InvertingGANs_with_ConsecutiveImgs
## HeadGAN: One-shot Neural Head Synthesis and Editing
- Paper：https://arxiv.org/abs/2012.08261
## Orthogonal Jacobian Regularization for Unsupervised Disentanglement in Image Generation
- Code：https://github.com/csyxwei/OroJaR
## Sketch Your Own GAN
- Paper：https://arxiv.org/abs/2108.02774
- Code：https://github.com/PeterWang512/GANSketching
## A Latent Transformer for Disentangled Face Editing in Images and Videos
- Paper：https://arxiv.org/abs/2106.11895
- Code：https://github.com/InterDigitalInc/Latent-Transformer
## Learning Facial Representations from the Cycle-consistency of Face
- Paper：https://arxiv.org/abs/2108.03427
- Code：https://github.com/jiarenchang/facecycle
## StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery
- Paper：https://arxiv.org/abs/2103.17249
- Code：https://github.com/orpatashnik/StyleCLIP

<a name="3.图像风格迁移"></a>
# 3.图像风格迁移（Image Transfer）
## ALADIN: All Layer Adaptive Instance Normalization for Fine-grained Style Similarity
- Paper：https://arxiv.org/abs/2103.09776
## Domain Aware Universal Style Transfer
- Paper：https://arxiv.org/abs/2108.04441
- Code：https://github.com/Kibeom-Hong/Domain-Aware-Style-Transfer
## AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer
- Paper：https://arxiv.org/abs/2108.03647
- Code：https://github.com/Huage001/AdaAttN

<a name="4.图像翻译"></a>
# 4.图像翻译（Image to Image Translation）
## SPatchGAN: A Statistical Feature Based Discriminator for Unsupervised Image-to-Image Translation
- Paper：https://arxiv.org/abs/2103.16219
- Code：https://github.com/NetEase-GameAI/SPatchGAN
## Scaling-up Disentanglement for Image Translation
- Paper：https://arxiv.org/abs/2103.14017
- Code：https://github.com/avivga/overlord

<a name="5.图像修复"></a>
# 5.图像修复（Image Inpaiting/Image Completion）
## Implicit Internal Video Inpainting
- Code：https://github.com/Tengfei-Wang/Implicit-Internal-Video-Inpainting
## Internal Video Inpainting by Implicit Long-range Propagation
- Code：https://github.com/Tengfei-Wang/Annotated-4K-Videos
## Occlusion-Aware Video Object Inpainting
- Paper：https://arxiv.org/abs/2108.06765
## High-Fidelity Pluralistic Image Completion with Transformers
- Paper：https://arxiv.org/abs/2103.14031
- Code：https://github.com/raywzy/ICT
## Image Inpainting via Conditional Texture and Structure Dual Generation
- Paper：https://arxiv.org/abs/2108.09760v1
- Code：https://github.com/Xiefan-Guo/CTSDG

<a name="6.图像超分辨率"></a>
# 6.图像超分辨率（Image Super-Resolution）
## Mutual Affine Network for Spatially Variant Kernel Estimation in Blind Image Super-Resolution
- Code：https://github.com/JingyunLiang/MANet
## Hierarchical Conditional Flow: A Unified Framework for Image Super-Resolution and Image Rescaling
- Code：https://github.com/JingyunLiang/HCFlow
## Deep Blind Video Super-resolution
- Code：https://github.com/csbhr/Deep-Blind-VSR
## Omniscient Video Super-Resolution
- Code：https://github.com/psychopa4/OVSR
## Learning A Single Network for Scale-Arbitrary Super-Resolution
- Paper：https://arxiv.org/abs/2004.03791
- Code：https://github.com/LongguangWang/ArbSR
## Deep Reparametrization of Multi-Frame Super-Resolution and Denoising
- Paper：https://arxiv.org/abs/2108.08286
## Lucas-Kanade Reloaded: End-to-End Super-Resolution from Raw Image Bursts
- Paper：https://arxiv.org/abs/2104.06191

<a name="7.其他底层视觉任务"></a>
# 7.其他底层视觉任务（Other Low Level Vision）
## **Overfitting the Data: Compact Neural Video Delivery via Content-aware Feature Modulation**
- Code：https://github.com/Anonymous-iccv2021-paper3163/CaFM-Pytorch
- 视频传输
## XVFI: eXtreme Video Frame Interpolation
- Paper：https://arxiv.org/abs/2103.16206
- Code：https://github.com/JihyongOh/XVFI
## Asymmetric Bilateral Motion Estimation for Video Frame Interpolation
- Paper： https://arxiv.org/abs/2108.06815
- Code： https://github.com/JunHeum/ABME
## Focal Frequency Loss for Image Reconstruction and Synthesis
- Paper：https://arxiv.org/abs/2012.12821
- Code：https://github.com/EndlessSora/focal-frequency-loss
- 频域损失，补充空域损失的不足
## ALL Snow Removed: Single Image Desnowing Algorithm Using Hierarchical Dual-tree Complex Wavelet Representation and Contradict Channel Loss
- Code：https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet
## Structure-Preserving Deraining with Residue Channel Prior Guidance
- Code：https://github.com/Joyies/SPDNet
## IICNet: A Generic Framework for Reversible Image Conversion
- Code：https://github.com/felixcheng97/IICNet
## Self-Conditioned Probabilistic Learning of Video Rescaling
- Paper：https://arxiv.org/abs/2107.11639
## HDR Video Reconstruction: A Coarse-to-fine Network and A Real-world Benchmark Dataset
- Paper：https://arxiv.org/abs/2103.14943
- Code：https://github.com/guanyingc/DeepHDRVideo
## Bringing Events into Video Deblurring with Non consecutively Blurry Frames
- Code：https://github.com/shangwei5/D2Net
## Rethinking Coarse-to-Fine Approach in Single Image Deblurring
- Paper：https://arxiv.org/abs/2108.05054
- Code：https://github.com/chosj95/MIMO-UNet
## Gap-closing Matters: Perceptual Quality Assessment and Optimization of Low-Light Image Enhancement
- Code：https://github.com/Baoliang93/Gap-closing-Matters
## SSH: A Self-Supervised Framework for Image Harmonization
- Paper：https://arxiv.org/abs/2108.06805
- Code：https://github.com/VITA-Group/SSHarmonization
## MUSIQ: Multi-scale Image Quality Transformer
- Paper：https://arxiv.org/abs/2108.05997
## Extending Neural P-frame Codecs for B-frame Coding
- Paper：https://arxiv.org/abs/2104.00531
## Towards Vivid and Diverse Image Colorization with Generative Color Prior
- Paper：https://arxiv.org/abs/2108.08826
## Towards Flexible Blind JPEG Artifacts Removal 
- Paper：https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/FBCNN_ICCV2021.pdf
- Code：https://github.com/jiaxi-jiang/FBCNN
