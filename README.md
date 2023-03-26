# Awesome-Efficient-AI-for-Large-Scale-Models
Paper survey of efficient computation for large scale models.

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
  
  2022-CVPR-Patch Slimming for Efficient Vision Transformers
  
  2022-CVPR-MiniViT: Compressing Vision Transformers with Weight Multiplexing
  
  2022-ECCV-An Efficient Spatio-Temporal Pyramid Transformer for Action Detection


Self-slimmed Vision Transformer, Zhuofan Zong, Kunchang Li, Guanglu Song, Yali Wang, Yu Qiao, Biao Leng, Yu Liu, ICLR 2022



##   Quantization for Vision Transformers

[arxiv] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.

[NeurIPS] Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer. [qnn]
 
 [ECCV] PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization. [qnn]
  
  [ECCV] Patch Similarity Aware Data-Free Quantization for Vision Transformers. [qnn]
  
  [IJCAI] FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer. [qnn] [code] [71⭐]
    
   [arxiv] Q-ViT: Fully Differentiable Quantization for Vision Transformer [qnn]
   
   [NeurIPS] BiT: Robustly Binarized Multi-distilled Transformer. [bnn] [code] [42⭐]
  
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

Improved Knowledge Distillation via Teacher Assistant: Bridging the Gap Between Student and Teacher. Mirzadeh et al. arXiv:1902.03393

Search to Distill: Pearls are Everywhere but not the Eyes. Liu Yu et al. CVPR 2020

Reducing the Teacher-Student Gap via Spherical Knowledge Disitllation, Jia Guo, Minghao Chen, Yao Hu, Chen Zhu, Xiaofei He, Deng Cai, 2020

Decoupled Knowledge Distillation, Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, Jiajun Liang, CVPR 2022, code

Prune Your Model Before Distill It, Jinhyuk Park and Albert No, ECCV 2022, code



Knowledge Distillation from A Stronger Teacher. arXiv preprint arXiv:2205.10536.

Masked Generative Distillation, Zhendong Yang, Zhe Li, Mingqi Shao, Dachuan Shi, Zehuan Yuan, Chun Yuan, ECCV 2022, code


Curriculum Temperature for Knowledge Distillation, Zheng Li, Xiang Li, Lingfeng Yang, Borui Zhao, Renjie Song, Lei Luo, Jun Li, Jian Yang, AAAI 2023, code

Knowledge distillation: A good teacher is patient and consistent, Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, Alexander Kolesnikov, CVPR 2022

Knowledge Distillation with the Reused Teacher Classifier, Defang Chen, Jian-Ping Mei, Hailin Zhang, Can Wang, Yan Feng, Chun Chen, CVPR 2022



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



