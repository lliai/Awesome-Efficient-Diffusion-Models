# Awesome-Efficient-AI-for-Large-Scale-Models
Paper survey of efficient computation for large scale models (e.g. Transformers, Vision Transformers, etc).

##  Knowledge Distillation in Vision Transformers


Training data-efficient image transformers & distillation through attention. ICML2021

Co-advise: Cross inductive bias distillation.CVPR2022.

Tinyvit: Fast pretraining distillation for small vision
transformers. arXiv preprint arXiv:2207.10666.

Attention Probe: Vision Transformer Distillation in the Wild. ICASSP2022


Dear KD: Data-Efficient Early Knowledge Distillation for Vision Transformers. CVPR2022


Efficient vision transformers via fine-grained manifold distillation. NIPS2022


 Cross-Architecture Knowledge Distillation. arXiv preprint arXiv:2207.05273. ACCV2022


MiniViT: Compressing Vision Transformers with Weight Multiplexing. CVPR2022

ViTKD: Practical Guidelines for ViT feature knowledge distillation. arXiv 2022, code

##   Model stitching for Vision Transformers

Deep Model Reassembly. NeurIPS, 2022

Stitchable Neural Networks. CVPR 2023 Highlight


##   Pruning for Vision Transformers

2019-NIPS-Are sixteen heads really better than one?

2020.11-Long Range Arena: A Benchmark for Efficient Transformers

 2021-KDDw-Vision Transformer Pruning
 
 2021-TCPS-TPrune: Efficient transformer pruning for mobile devices
 
 2021.05-MLPruning: A Multilevel Structured Pruning Framework for Transformer-based Model [Code]
 

 2021.09-HFSP: A Hardware-friendly Soft Pruning Framework for Vision Transformers
    
 2021.11-Pruning Self-attentions into Convolutional Layers in Single Path [Code]
  
  2021.11-A Memory-saving Training Framework for Transformers [Code]


 2022-AAAI-Less is More: Pay Less Attention in Vision Transformers
 
 2022-ICLR-Unified Visual Transformer Compression
  
  2022-ICLR-Self-slimmed Vision Transformer
  
  2022-CVPR-Patch Slimming for Efficient Vision Transformers
  
  2022-CVPR-MiniViT: Compressing Vision Transformers with Weight Multiplexing
  
  2022-ECCV-An Efficient Spatio-Temporal Pyramid Transformer for Action Detection



##   Quantization for Vision Transformers

[arxiv] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.

[NeurIPS] Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer. [qnn]
 
 [ECCV] PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization. [qnn]
  
  [ECCV] Patch Similarity Aware Data-Free Quantization for Vision Transformers. [qnn]
  
  [IJCAI] FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer. [qnn] [code] [71]
    
   [arxiv] Q-ViT: Fully Differentiable Quantization for Vision Transformer [qnn]
   
   [NeurIPS] BiT: Robustly Binarized Multi-distilled Transformer. [bnn] [code] [42]
  
  [NeurIPS] ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers. [qnn]

   [NeurIPS] Post-Training Quantization for Vision Transformer. [mixed]
    
   [NeurIPS] Fully Quantized Transformer for Improved Translation. [qnn] [nlp]
   
   [ICML] Efficient 8-Bit Quantization of Transformer Neural Machine Language Translation Model. [qnn] [nlp]
   
   [IJCAI] Towards Fully 8-bit Integer Inference for the Transformer Model. [qnn] [nlp]
   
   [EMNLP] TernaryBERT: Distillation-aware Ultra-low Bit BERT. [qnn]
   
   [EMNLP] Fully Quantized Transformer for Machine Translation. [qnn]
   
   [ACL] On the Distribution, Sparsity, and Inference-time Quantization of Attention Values in Transformers. [qnn]
   
   [arxiv] Post-Training Quantization for Vision Transformer. [qnn]


##  Methods for Distillation Gaps

Improved Knowledge Distillation via Teacher Assistant: Bridging the Gap Between Student and Teacher. Mirzadeh et al. AAAI2020

Search to Distill: Pearls are Everywhere but not the Eyes. Liu Yu et al. CVPR 2020

Reducing the Teacher-Student Gap via Spherical Knowledge Disitllation, arXiv:2020

Knowledge Distillation via the Target-aware Transformer, CVPR2022

Decoupled Knowledge Distillation, Borui Zhao, et al. , CVPR 2022, code

Prune Your Model Before Distill It, Jinhyuk Park and Albert No, ECCV 2022, code

Asymmetric Temperature Scaling Makes Larger Networks Teach Well Again, NeurIPS 2022

Weighted Distillation with Unlabeled Examples, NeurIPS 2022

Respecting Transfer Gap in Knowledge Distillation, NeurIPS 2022


Knowledge Distillation from A Stronger Teacher. arXiv preprint arXiv:2205.10536.

Masked Generative Distillation, Zhendong Yang, et al. , ECCV 2022, code


Curriculum Temperature for Knowledge Distillation, Zheng Li, et al. , AAAI 2023, code

Knowledge distillation: A good teacher is patient and consistent, Lucas Beyeret al.  CVPR 2022

Knowledge Distillation with the Reused Teacher Classifier, Defang Chen, et al. , CVPR 2022

Scaffolding a Student to Instill Knowledge, ICLR2023

Function-Consistent Feature Distillation, ICLR2023

Better Teacher Better Student: Dynamic Prior Knowledge for Knowledge Distillation, ICLR2023

Supervision Complexity and its Role in Knowledge Distillation, ICLR2023

## Training-Free Neural & Transformer Architecture Search 

 Zero-Cost Proxies for Lightweight NAS
 
 Unifying and Boosting Gradient-Based Training-Free Neural Architecture Search
 
LiteTransformerSearch: Training-free On-device Search for Efficient Autoregressive Language Models

Training-free Transformer Architecture Search

Training-Free Hardware-Aware Neural Architecture Search with Reinforcement Learning

Understanding and Accelerating Neural Architecture Search with Training-Free and Theory-Grounded Metrics
    
A Feature Fusion Based Indicator for Training-Free Neural Architecture Search
    
Neural Architecture Search without Training
    
Zen-NAS: A Zero-Shot NAS for High-Performance Deep Image Recognition
    
Revisiting Efficient Object Detection Backbones from Zero-Shot Neural Architecture Search
    
A Training-Free Genetic Neural Architecture Search

EcoNAS: Finding Proxies for Economical Neural Architecture Search
    
EPE-NAS: Efficient Performance Estimation Without Training for Neural Architecture Search
    
How does topology influence gradient propagation and model performance of deep networks with DenseNet-type skip connections?
    
FLASH: Fast Neural Architecture Search with Hardware Optimization
    
Deep Architecture Connectivity Matters for Its Convergence: A Fine-Grained Analysis
    
 Reducing Neural Architecture Search Spaces with Training-Free Statistics and Computational Graph Clustering
    
EZNAS: Evolving Zero-Cost Proxies For Neural Architecture Scoring
    
Zero-Cost Proxies Meet Differentiable Architecture Search
    
Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective
    
Training-Free Multi-objective Evolutionary Neural Architecture Search via Neural Tangent Kernel and Number of Linear Regions
    
Extensible Proxy for Efficient NAS
    
ZiCo: Zero-shot NAS via Inverse Coefficient of Variation on Gradients



## General Transformer Search

| Title                                                                                                   | Venue         | Group                  |
|:--------------------------------------------------------------------------------------------------------|:--------------|:-----------------------|
| [LiteTransformerSearch: Training-free On-device Search for Efficient Autoregressive Language Models](https://arxiv.org/pdf/2203.02094.pdf) | arxiv [March'22]|  MSR |
| [Training Free Transformer Architecture Search](https://arxiv.org/abs/2203.12217)                    | **CVPR'22** |  Tencent & Xiamen University |
| [Searching the Search Space of Vision Transformer](https://proceedings.neurips.cc/paper/2021/file/48e95c45c8217961bf6cd7696d80d238-Paper.pdf)                    | **NeurIPS'21** | MSRA, Stony Brook University              |
| [UniNet: Unified Architecture Search with Convolutions, Transformer and MLP](https://arxiv.org/pdf/2110.04035.pdf)                              | arxiv [Oct'21] | SenseTime              |
| [Analyzing and Mitigating Interference in Neural Architecture Search](https://arxiv.org/pdf/2108.12821.pdf)                              | arxiv [Aug'21] | Tsinghua, MSR         |
| [BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search](https://arxiv.org/pdf/2103.12424.pdf) | **ICCV'21**       | Sun Yat-sen University |
| [Memory-Efficient Differentiable Transformer Architecture Search](https://aclanthology.org/2021.findings-acl.372.pdf)                              | **ACL-IJCNLP'21** | MSR, Peking University              |
| [Finding Fast Transformers: One-Shot Neural Architecture Search by Component Composition](https://arxiv.org/pdf/2008.06808.pdf)                 | arxiv [Aug'20] | Google Research        |
| [AutoTrans: Automating Transformer Design via Reinforced Architecture Search](https://arxiv.org/pdf/2009.02070.pdf)                             | arxiv [Sep'20] | Fudan University       |
| [NASABN: A Neural Architecture Search Framework for Attention-Based Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9207600)                                                                                 | **IJCNN'20**       | Chinese Academy of Sciences           |
| [NAT: Neural Architecture Transformer for Accurate and Compact Architectures](https://arxiv.org/pdf/1910.14488.pdf)                             | **NeurIPS'19**    | Tencent AI             |
| [The Evolved Transformer](http://proceedings.mlr.press/v97/so19a/so19a.pdf)                                                                                 | **ICML'19**       | Google Brain           |


## Domain Specific Transformer Search
### Vision

| Title                                                                                                   | Venue         | Group                  |
|:--------------------------------------------------------------------------------------------------------|:--------------|:-----------------------|
| [𝛼NAS: Neural Architecture Search using Property Guided Synthesis](https://arxiv.org/abs/2205.03960)        | arxiv |  MIT, Google  |
| [NASViT: Neural Architecture Search for Efficient Vision Transformers with Gradient Conflict aware Supernet Training](https://openreview.net/pdf?id=Qaw16njk6L)        | **ICLR'22** |  Meta Reality Labs    |
| [AutoFormer: Searching Transformers for Visual Recognition](https://arxiv.org/pdf/2107.00651.pdf)                              | **ICCV'21** | MSR              |
| [GLiT: Neural Architecture Search for Global and Local Image Transformer](https://arxiv.org/pdf/2107.02960.pdf) | **ICCV'21**       | University of Sydney |
| [Searching for Efficient Multi-Stage Vision Transformers](https://neural-architecture-ppf.github.io/papers/00011.pdf)                             | ICCV'21 workshop | MIT       |
| [HR-NAS: Searching Efficient High-Resolution Neural Architectures with Lightweight Transformers](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_HR-NAS_Searching_Efficient_High-Resolution_Neural_Architectures_With_Lightweight_Transformers_CVPR_2021_paper.pdf)                             | **CVPR'21**    | Bytedance Inc.             |
| [ViTAS: Vision Transformer Architecture Search](https://arxiv.org/pdf/2106.13700.pdf)                 | arxiv [June'21] | SenseTime, Tsingua University        |


### Natural Language Processing

| Title                                                                                                   | Venue         | Group                  |
|:--------------------------------------------------------------------------------------------------------|:--------------|:-----------------------|
| [AutoBERT-Zero: Evolving the BERT backbone from scratch](https://arxiv.org/pdf/2107.07445.pdf) | **AAAI'22** | Huawei Noah’s Ark Lab       |
| [Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/pdf/2109.08668.pdf)  | **NeurIPS'21**    | Google           |
| [AutoTinyBERT: Automatic Hyper-parameter Optimization for Efficient Pre-trained Language Models](https://aclanthology.org/2021.acl-long.400.pdf)  | **ACL'21**    | Tsinghua, Huawei Naoh's Ark    |
| [NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search](https://arxiv.org/pdf/2105.14444.pdf) | **KDD'21**       | MSR, Tsinghua University |
| [HAT: Hardware-Aware Transformers for Efficient Natural Language Processing](https://arxiv.org/pdf/2005.14187.pdf)  | **ACL'20**    | MIT           |

### Automatic Speech Recognition

| Title                                                                                                   | Venue         | Group                  |
|:--------------------------------------------------------------------------------------------------------|:--------------|:-----------------------|
| [LightSpeech: Lightweight and Fast Text to Speech with Neural Architecture Search](https://arxiv.org/abs/2102.04040) | **ICASSP'21** | MSR       |
| [Efficient Gradient-Based Neural Architecture Search For End-to-End ASR](https://dl.acm.org/doi/abs/10.1145/3461615.3491109) | ICMI-MLMI'21 | NPU, Xi'an       |
| [Improved Conformer-based End-to-End Speech Recognition Using Neural Architecture Search](https://arxiv.org/pdf/2104.05390.pdf) | arxiv [April'21]  | Chinese Academy of Sciences |
| [Evolved Speech-Transformer: Applying Neural Architecture Search to End-to-End Automatic Speech Recognition](https://indico2.conference4me.psnc.pl/event/35/contributions/3122/attachments/301/324/Tue-1-8-5.pdf)  | **INTERSPEECH'20**    | VUNO Inc.           |


### Transformers Knowledge: Insights, Searchable parameters, Attention

| Title                                                                                                   | Venue         | Group                  |
|:--------------------------------------------------------------------------------------------------------|:--------------|:-----------------------|
| [Seperable Self Attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680)  | arxiv'22 | Apple |
| [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)  | arxiv'22 | Snap Inc |
| [Neighborhood Attention Transformer](https://arxiv.org/pdf/2204.07143.pdf)  | arxiv'22 | Meta AI |
| [Training Compute Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)  | arxiv'22 | DeepMind |
| [Parameter-efficient Fine-tuning for Vision Transformers](https://arxiv.org/pdf/2203.16329.pdf)  | arxiv | MSR & UCSC |
| [CMT: Convolutional Neural Networks meet Vision Transformers](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_CMT_Convolutional_Neural_Networks_Meet_Vision_Transformers_CVPR_2022_paper.html)  | **CVPR'22** | Huawei Noah’s Ark Lab |
| [Patch Slimming for Efficient Vision Transformers](https://openaccess.thecvf.com/content/CVPR2022/html/Tang_Patch_Slimming_for_Efficient_Vision_Transformers_CVPR_2022_paper.html)  | **CVPR'22** | Huawei Noah’s Ark Lab |
| [Lite Vision Transformer with Enhanced Self-Attention](https://arxiv.org/abs/2112.10809)  | **CVPR'22** | Johns Hopkins University, Adobe |
| [TubeDETR: Spatio-Temporal Video Grounding with Transformers](https://arxiv.org/pdf/2203.16434.pdf)  | **CVPR'22 (Oral)** | CNRS & Inria |
| [Beyond Fixation: Dynamic Window Visual Transformer](https://arxiv.org/abs/2203.12856)  | **CVPR'22** | UT Sydney & RMIT University |
| [BEiT: BERT Pre-Training of Image Transformers](https://openreview.net/forum?id=p-BhZSz59o4)  | **ICLR'22 (Oral)** | MSR |
| [How Do Vision Transformers Work?](https://openreview.net/forum?id=D78Go4hVcxO)  | **ICLR'22 (Spotlight)** | NAVER AI |
| [Scale Efficiently: Insights from Pretraining and FineTuning Transformers](https://openreview.net/pdf?id=f2OYVDyfIB)  | **ICLR'22** | Google Research |
| [Tuformer: Data-Driven Design of Expressive Transformer by Tucker Tensor Representation](https://openreview.net/pdf?id=V0A5g83gdQ_)  | **ICLR'22** | UoMaryland |
| [DictFormer: Tiny Transformer with Shared Dictionary](https://openreview.net/pdf?id=GWQWAeE9EpB)  | **ICLR'22** | Samsung Research |
| [QuadTree Attention for Vision Transformers](https://arxiv.org/pdf/2201.02767.pdf)  | **ICLR'22** | Alibaba AI Lab |
| [Expediting Vision Transformers via Token Reorganization](https://openreview.net/pdf?id=BjyvwnXXVn_)  | **ICLR'22 (Spotlight)** | UC San Diego & Tencent AI Lab |
| [UniFormer: Unified Transformer for Efficient Spatial-Temporal Representation Learning](https://openreview.net/forum?id=nBU_u6DLvoK)  | arxiv | - |
| [Patches are All You Need ?](https://openreview.net/pdf?id=TVHS5Y4dNvM)                              | arxiv'22 | - |
| [Hierarchical Transformers Are More Efficient Language Models](https://arxiv.org/pdf/2110.13711.pdf)       | arxiv'21 | Google Research, UoWarsaw |
| [Transformer in Transformer](https://papers.nips.cc/paper/2021/file/854d9fca60b4bd07f9bb215d59ef5561-Paper.pdf)    | **NeurIPS'21** | Huawei Noah's Ark |
| [Long-Short Transformer: Efficient Transformers for Language and Vision](https://papers.nips.cc/paper/2021/file/9425be43ba92c2b4454ca7bf602efad8-Paper.pdf)  | **NeurIPS'21** | NVIDIA |
| [Memory-efficient Transformers via Top-k Attention](https://aclanthology.org/2021.sustainlp-1.5.pdf)  | EMNLP Workshop '21 | Allen AI |
| [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)                              | **ICCV'21 best paper** | MSR |
| [Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/pdf/2103.16302.pdf)                              | **ICCV'21** | NAVER AI |
| [What makes for hierarchical vision transformers](https://arxiv.org/pdf/2107.02174.pdf)                                                                                 | arxiv [Sept'21]       | HUST           |
| [AutoAttend: Automated Attention Representation Search](http://proceedings.mlr.press/v139/guan21a/guan21a.pdf) | **ICML'21**       | Tsinghua University |
| [Rethinking Attention with Performers](https://openreview.net/pdf?id=Ua6zuk0WRH)                              | **ICLR'21 Oral** | Google              |
| [LambdaNetworks: Modeling long-range Interactions without Attention](https://openreview.net/forum?id=xTJEN-ggl1b)                 | **ICLR'21** | Google Research        |
| [HyperGrid Transformers](https://openreview.net/pdf?id=hiq1rHO8pNT)                             | **ICLR'21** | Google Research       |
| [LocalViT: Bringing Locality to Vision Transformers](https://arxiv.org/pdf/2104.05707.pdf)                             | arxiv [April'21]   | ETH Zurich            |
| [Compressive Transformers for Long Range Sequence Modelling](https://openreview.net/forum?id=SylKikSYDH)                                                                                 | **ICLR'20**       | DeepMind     |
| [Improving Transformer Models by Reordering their Sublayers](https://arxiv.org/pdf/1911.03864.pdf)                                                                                 | **ACL'20**       | FAIR, Allen AI           |
| [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://www.aclweb.org/anthology/P19-1580.pdf)                                                                                 | **ACL'19**       | Yandex           |

## Transformer Surveys
| Title                                                                                                   | Venue         | Group                  |
|:--------------------------------------------------------------------------------------------------------|:--------------|:-----------------------|
| [Transformers in Vision: A Survey](https://arxiv.org/pdf/2101.01169.pdf) | arxiv [Oct'21] | MBZ University of AI      |
[Neural Architecture Search for Transformers: A Survey](https://ieeexplore.ieee.org/document/9913476) | IEEE xplore [Sep'22] | Iowa State Uni |
| [A Survey of Visual Transformers](https://arxiv.org/pdf/2111.06091.pdf) | arxiv [Nov'21]  | CAS |
| [Efficient Transformers: A Survey](https://arxiv.org/pdf/2009.06732.pdf) | arxiv [Sept'21]  | Google Research |

### Misc resources
- [Awesome Visual Transformer](https://github.com/dk-liang/Awesome-Visual-Transformer)
- [Vision Transformer & Attention Awesome List](https://github.com/cmhungsteve/Awesome-Transformer-Attention)


