# Awesome-ICCV2021-Low-Level-Vision[![Awesome](https://camo.githubusercontent.com/13c4e50d88df7178ae1882a203ed57b641674f94/68747470733a2f2f63646e2e7261776769742e636f6d2f73696e647265736f726875732f617765736f6d652f643733303566333864323966656437386661383536353265336136336531353464643865383832392f6d656469612f62616467652e737667)](https://github.com/sindresorhus/awesome)
A Collection of Papers and Codes for ICCV2021 Low Level Vision and Image Generation

整理汇总下2021年ICCV中图像生成（Image Generation）和底层视觉(Low-Level Vision)任务相关的论文和代码，包括图像生成，图像编辑，图像风格迁移，图像翻译，图像修复，图像超分及其他底层视觉任务。大家如果觉得有帮助，欢迎star~~

**参考或转载请注明出处,文中有不足或者需要补充的地方也欢迎PR**

ICCV2021官网：[https://iccv2021.thecvf.com/](https://iccv2021.thecvf.com/)

ICCV2021完整论文列表：[https://openaccess.thecvf.com/ICCV2021](https://openaccess.thecvf.com/ICCV2021)

开会时间：2021年10月11日-10月17日


**【Contents】**
- [1.图像生成（Image Generation）](#1.图像生成)
- [2.图像编辑（Image Manipulation/Image Editing）](#2.图像编辑)
- [3.图像风格迁移（Image Transfer）](#3.图像风格迁移)
- [4.图像翻译（Image to Image Translation）](#4.图像翻译)
- [5.图像修复（Image Inpaiting/Image Completion）](#5.图像修复)
- [6.图像超分辨率（Image Super-Resolution）](#6.图像超分辨率)
- [7.图像去雨（Image Deraining）](#7.图像去雨)
- [8.图像去雾（Image Dehazing）](#8.图像去雾)
- [9.图像去模糊（Image Deblurring）](#9.去模糊)
- [10.图像去噪（Image Denoising）](#10.去噪)
- [11.图像恢复（Image Restoration）](#11.图像恢复)
- [12.图像增强（Image Enhancement）](#12.图像增强)
- [13.图像质量评价（Image Quality Assessment）](#13.图像质量评价)
- [14.插帧（Frame Interpolation）](#14.插帧)
- [15.视频/图像压缩（Video/Image Compression）](#15.视频压缩)
- [16.其他底层视觉任务（Other Low Level Vision）](#16.其他底层视觉任务)


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
## The Animation Transformer: Visual Correspondence via Segment Matching
- Paper：https://arxiv.org/abs/2109.02614
- 手绘图变动画
## Image Synthesis via Semantic Composition
- Paper：https://shepnerd.github.io/scg/resources/01145.pdf
- Code：https://github.com/dvlab-research/SCGAN
## Detail Me More: Improving GAN's Photo-Realism of Complex Scenes
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Gadde_Detail_Me_More_Improving_GANs_Photo-Realism_of_Complex_Scenes_ICCV_2021_paper.html
## De-Rendering Stylized Texts
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Shimoda_De-Rendering_Stylized_Texts_ICCV_2021_paper.html
- Code：https://github.com/dvlab-research/SCGAN


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
## Talk-to-Edit: Fine-Grained Facial Editing via Dialog
- Paper：https://arxiv.org/abs/2109.04425
- Code：https://github.com/yumingj/Talk-to-Edit
## Dressing in Order: Recurrent Person Image Generation for Pose Transfer, Virtual Try-on and Outfit Editing
- Paper：https://cuiaiyu.github.io/dressing-in-order/Cui_Dressing_in_Order.pdf
- Code：https://github.com/cuiaiyu/dressing-in-order
## GAN-Control: Explicitly Controllable GANs
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Shoshan_GAN-Control_Explicitly_Controllable_GANs_ICCV_2021_paper.html
- Code：https://github.com/cuiaiyu/dressing-in-order
## Explaining in Style: Training a GAN To Explain a Classifier in StyleSpace
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Lang_Explaining_in_Style_Training_a_GAN_To_Explain_a_Classifier_ICCV_2021_paper.html
- Code：https://github.com/google/explaining-in-style


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
## Diverse Image Style Transfer via Invertible Cross-Space Mapping
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Diverse_Image_Style_Transfer_via_Invertible_Cross-Space_Mapping_ICCV_2021_paper.html
## StyleFormer: Real-Time Arbitrary Style Transfer via Parametric Style Composition
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Wu_StyleFormer_Real-Time_Arbitrary_Style_Transfer_via_Parametric_Style_Composition_ICCV_2021_paper.html

<a name="4.图像翻译"></a>
# 4.图像翻译（Image to Image Translation）
## SPatchGAN: A Statistical Feature Based Discriminator for Unsupervised Image-to-Image Translation
- Paper：https://arxiv.org/abs/2103.16219
- Code：https://github.com/NetEase-GameAI/SPatchGAN
## Scaling-up Disentanglement for Image Translation
- Paper：https://arxiv.org/abs/2103.14017
- Code：https://github.com/avivga/overlord
## Unaligned Image-to-Image Translation by Learning to Reweight
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Xie_Unaligned_Image-to-Image_Translation_by_Learning_to_Reweight_ICCV_2021_paper.html
- Code：https://github.com/Mid-Push/IrwGAN
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
## CR-Fill: Generative Image Inpainting with Auxiliary Contextual Reconstruction
- Paper：https://arxiv.org/abs/2011.12836
- Code：https://github.com/zengxianyu/crfill
## FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Liu_FuseFormer_Fusing_Fine-Grained_Information_in_Transformers_for_Video_Inpainting_ICCV_2021_paper.html
- Code：https://github.com/ruiliu-ai/FuseFormer


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
## Attention-Based Multi-Reference Learning for Image Super-Resolution
- Paper：https://openaccess.thecvf.com/content/ICCV2021/papers/Pesavento_Attention-Based_Multi-Reference_Learning_for_Image_Super-Resolution_ICCV_2021_paper.pdf
## Fourier Space Losses for Efficient Perceptual Image Super-Resolution
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Fuoli_Fourier_Space_Losses_for_Efficient_Perceptual_Image_Super-Resolution_ICCV_2021_paper.html
## COMISR: Compression-Informed Video Super-Resolution
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Li_COMISR_Compression-Informed_Video_Super-Resolution_ICCV_2021_paper.html
- Code：https://github.com/google-research/google-research/tree/master/comisr
- 针对压缩后的视频超分
## Designing a Practical Degradation Model for Deep Blind Image Super-Resolutio
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Designing_a_Practical_Degradation_Model_for_Deep_Blind_Image_Super-Resolution_ICCV_2021_paper.html
- Code：https://github.com/cszn/BSRGAN
## Event Stream Super-Resolution via Spatiotemporal Constraint Learning
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Li_Event_Stream_Super-Resolution_via_Spatiotemporal_Constraint_Learning_ICCV_2021_paper.html
## Super-Resolving Cross-Domain Face Miniatures by Peeking at One-Shot Exemplar
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Li_Super-Resolving_Cross-Domain_Face_Miniatures_by_Peeking_at_One-Shot_Exemplar_ICCV_2021_paper.html


<a name="7.图像去雨"></a>
# 7.图像去雨（Image Deraining）
## Structure-Preserving Deraining with Residue Channel Prior Guidance
- Code：https://github.com/Joyies/SPDNet
## Improving De-Raining Generalization via Neural Reorganization
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Xiao_Improving_De-Raining_Generalization_via_Neural_Reorganization_ICCV_2021_paper.html
- Code：https://github.com/cszn/BSRGAN
## Unpaired Learning for Deep Image Deraining With Rain Direction Regularizer
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Unpaired_Learning_for_Deep_Image_Deraining_With_Rain_Direction_Regularizer_ICCV_2021_paper.html
- Code：https://github.com/cszn/BSRGAN

<a name="8.图像去雾"></a>
# 8.图像去雾（Image Dehazing）

<a name="9.去模糊"></a>
# 9.图像去模糊（Image Deblurring）
## Bringing Events into Video Deblurring with Non consecutively Blurry Frames
- Code：https://github.com/shangwei5/D2Net
## Rethinking Coarse-to-Fine Approach in Single Image Deblurring
- Paper：https://arxiv.org/abs/2108.05054
- Code：https://github.com/chosj95/MIMO-UNet
## Bringing Events into Video Deblurring with Non consecutively Blurry Frames 
- Code：https://github.com/shangwei5/D2Net
## Single Image Defocus Deblurring Using Kernel-Sharing Parallel Atrous Convolutions
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Son_Single_Image_Defocus_Deblurring_Using_Kernel-Sharing_Parallel_Atrous_Convolutions_ICCV_2021_paper.html

<a name="10.去噪"></a>
# 10.图像去噪（Image Denoising）
## C2N: Practical Generative Noise Modeling for Real-World Denoising
- Paper：https://openaccess.thecvf.com/content/ICCV2021/papers/Jang_C2N_Practical_Generative_Noise_Modeling_for_Real-World_Denoising_ICCV_2021_paper.pdf
## Self-Supervised Image Prior Learning With GMM From a Single Noisy Image
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Self-Supervised_Image_Prior_Learning_With_GMM_From_a_Single_Noisy_ICCV_2021_paper.html
- Code：https://github.com/HUST-Tan/SS-GMM


<a name="11.图像恢复"></a>
# 11.图像恢复（Image Restoration）
## Spatially-Adaptive Image Restoration using Distortion-Guided Networks
- Paper：https://arxiv.org/abs/2108.08617
## Dynamic Attentive Graph Learning for Image Restoration
- Paper：https://arxiv.org/abs/2109.06620
- Code：https://github.com/jianzhangcs/DAGL

<a name="12.图像增强"></a>
# 12.图像增强（Image Enhancement）
## StarEnhancer: Learning Real-Time and Style-Aware Image Enhancement
- Paper：https://arxiv.org/abs/2107.12898
- Code：https://github.com/IDKiro/StarEnhancer
## Real-time Image Enhancer via Learnable Spatial-aware 3D Lookup Tables
- Paper：https://arxiv.org/abs/2108.08697
## Representative Color Transform for Image Enhancement
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Representative_Color_Transform_for_Image_Enhancement_ICCV_2021_paper.html
## Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Zheng_Adaptive_Unfolding_Total_Variation_Network_for_Low-Light_Image_Enhancement_ICCV_2021_paper.html
- Code：https://github.com/YU-Zhiyang/WEVI

<a name="13.图像质量评价"></a>
# 13.图像质量评价（Image Quality Assessment）
## MUSIQ: Multi-scale Image Quality Transformer
- Paper：https://arxiv.org/abs/2108.05997

<a name="14.插帧"></a>
# 14.插帧（Frame Interpolation）
## XVFI: eXtreme Video Frame Interpolation
- Paper：https://arxiv.org/abs/2103.16206
- Code：https://github.com/JihyongOh/XVFI
## Asymmetric Bilateral Motion Estimation for Video Frame Interpolation
- Paper： https://arxiv.org/abs/2108.06815
- Code： https://github.com/JunHeum/ABME
## Training Weakly Supervised Video Frame Interpolation With Events
- Paper： https://openaccess.thecvf.com/content/ICCV2021/html/Yu_Training_Weakly_Supervised_Video_Frame_Interpolation_With_Events_ICCV_2021_paper.html


<a name="15.视频压缩"></a>
# 15.视频/图像压缩（Video/Image Compression）
## Extending Neural P-frame Codecs for B-frame Coding
- Paper：https://arxiv.org/abs/2104.00531
## Variable-Rate Deep Image Compression through Spatially-Adaptive Feature Transform
- Paper：https://arxiv.org/abs/2108.09551
- Code：https://github.com/micmic123/QmapCompression
## Efficient Video Compression via Content-Adaptive Super-Resolution
- Paper：https://openaccess.thecvf.com/content/ICCV2021/papers/Khani_Efficient_Video_Compression_via_Content-Adaptive_Super-Resolution_ICCV_2021_paper.pdf
- Code：https://github.com/AdaptiveVC/SRVC


<a name="16.其他底层视觉任务"></a>
# 16.其他底层视觉任务（Other Low Level Vision）
## Overfitting the Data: Compact Neural Video Delivery via Content-aware Feature Modulation
- Code：https://github.com/Anonymous-iccv2021-paper3163/CaFM-Pytorch
- 视频传输
## Focal Frequency Loss for Image Reconstruction and Synthesis
- Paper：https://arxiv.org/abs/2012.12821
- Code：https://github.com/EndlessSora/focal-frequency-loss
- 频域损失，补充空域损失的不足
## ALL Snow Removed: Single Image Desnowing Algorithm Using Hierarchical Dual-tree Complex Wavelet Representation and Contradict Channel Loss
- Code：https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet
## IICNet: A Generic Framework for Reversible Image Conversion
- Code：https://github.com/felixcheng97/IICNet
## Self-Conditioned Probabilistic Learning of Video Rescaling
- Paper：https://arxiv.org/abs/2107.11639
## HDR Video Reconstruction: A Coarse-to-fine Network and A Real-world Benchmark Dataset
- Paper：https://arxiv.org/abs/2103.14943
- Code：https://github.com/guanyingc/DeepHDRVideo
## A New Journey from SDRTV to HDRTV
- Paper：https://arxiv.org/abs/2108.07978
- Code：https://github.com/chxy95/HDRTVNet
## SSH: A Self-Supervised Framework for Image Harmonization
- Paper：https://arxiv.org/abs/2108.06805
- Code：https://github.com/VITA-Group/SSHarmonization
## Towards Vivid and Diverse Image Colorization with Generative Color Prior
- Paper：https://arxiv.org/abs/2108.08826
## Towards Flexible Blind JPEG Artifacts Removal 
- Paper：https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/FBCNN_ICCV2021.pdf
- Code：https://github.com/jiaxi-jiang/FBCNN
### Location-Aware Single Image Reflection Removal
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Dong_Location-Aware_Single_Image_Reflection_Removal_ICCV_2021_paper.html
- Code：https://github.com/zdlarr/Location-aware-SIRR
## Learning To Remove Refractive Distortions From Underwater Images
- Paper：https://openaccess.thecvf.com/content/ICCV2021/html/Thapa_Learning_To_Remove_Refractive_Distortions_From_Underwater_Images_ICCV_2021_paper.html

# 相关Low-Level-Vision整理
- [Awesome-CVPR2021/CVPR2020-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-CVPR2021-CVPR2020-Low-Level-Vision)
- [Awesome-ECCV2020-Low-Level-Vision](https://github.com/Kobaayyy/Awesome-ECCV2020-Low-Level-Vision)
