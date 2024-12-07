# Awesome-Efficient-Diffusion-Models

This repository provides list of research papers, resources, and code related to efficient diffusion models, including knowledge distillation, pruning, quantization, and other optimization techniques.

## Knowledge Distillation for Diffusion Models

Knowledge distillation is a technique used to transfer knowledge from a larger "teacher" model to a smaller "student" model, enabling efficient inference and deployment. This section covers various approaches for knowledge distillation applied to diffusion models.

### Papers

- [CVPR] On Distillation of Guided Diffusion Models
- [ICME] Accelerating Diffusion Sampling with Classifier-based Feature Distillation
- [ICML] Accelerating Diffusion-based Combinatorial Optimization Solvers by Progressive Distillation
- [ICML] Towards Safe Self-Distillation of Internet-Scale Text-to-Image Diffusion Models 
- [arxiv] BOOT: Data-free Distillation of Denoising Diffusion Models with Bootstrapping
- [arxiv] SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds
- [arxiv] Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling 
- [arxiv] Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed
- [arxiv] On Architectural Compression of Text-to-Image Diffusion Models
- [arxiv] Progressive Distillation for Fast Sampling of Diffusion Models
- [ICML] Consistency Models
- [ICLR] Improved Techniques for Training Consistency Models
- [arxiv] Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation
- [arxiv] SCott: Accelerating Diffusion Models with Stochastic Consistency Distillation
- [arxiv] Towards a mathematical theory for consistency training in diffusion models
- [arxiv] Linear Combination of Saved Checkpoints Makes Consistency and Diffusion Models Better
- [arxiv] Adversarial Diffusion Distillation
- [arxiv] Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis
- [arxiv] Bidirectional Consistency Models
- [arxiv] Your Student is Better Than Expected: Adaptive Teacher-Student Collaboration for Text-Conditional Diffusion Models
- [ICLR] Consistency Trajectory Model
- [arxiv] Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference.
- [arxiv] VideoLCM: Video Latent Consistency Model
- [arxiv] Clockwork Diffusion: Efficient Generation With Model-Step Distillation
- [arxiv]  LoRA-Enhanced Distillation on Guided Diffusion Models
- [arxiv] One-step Diffusion with Distribution Matching Distillation
- [arxiv] SDXL-Lightning: Progressive Adversarial Diffusion Distillation
- [arxiv] SwiftBrush : One-Step Text-to-Image Diffusion Model with Variational Score Distillation

## Pruning and Sparsity for Diffusion Models

Pruning and sparsity techniques aim to reduce the computational complexity and memory footprint of diffusion models by removing redundant parameters or activations. This section covers various pruning and sparsity methods applied to diffusion models.

### Papers

- [arxiv] Token Merging for Fast Stable Diffusion
- [arxiv] ToDo: Token Downsampling for Efficient Generation of High-Resolution Images
- [NeurIPS] Diff-Pruning: Structural Pruning for Diffusion Models
- [ICLR]  Denoising Diffusion Step-aware Models
- [CVPR]  DeepCache: Accelerating Diffusion Models for Free
- [arxiv] FRDiff: Feature Reuse for Exquisite Zero-shot Acceleration of Diffusion Models
- [arxiv] F3-Pruning: A Training-Free and Generalized Pruning Strategy towards Faster and Finer Text-to-Video Synthesis
- [arxiv] Cache Me if You Can: Accelerating Diffusion Models through Block Caching
- [arxiv] Faster Diffusion: Rethinking the Role of UNet Encoder in Diffusion Models
- [arxiv] DeeDiff: Dynamic Uncertainty-Aware Early Exiting for Accelerating Diffusion Model Generation
- [arxiv] T-Stitch: Accelerating Sampling in Pre-Trained Diffusion Models with Trajectory Stitching
- [arxiv] Cross-Attention Makes Inference Cumbersome in Text-to-Image Diffusion Models
- [arxiv] LAPTOP-Diff: Layer Pruning and Normalized Distillation for Compressing Diffusion Models
- [TPAMI] Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models

## Quantization for Diffusion Models

Quantization techniques aim to reduce the precision of model parameters and activations, enabling smaller model sizes and faster inference. This section covers various quantization methods applied to diffusion models.

### Papers

- [ArXiv] BinaryDM: Towards Accurate Binarization of Diffusion Model 
- [ICLR] Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning
- [CVPR] Post-training Quantization on Diffusion Models 
- [CVPR] Regularized Vector Quantization for Tokenized Image Synthesis
- [ICCV] Q-Diffusion: Quantizing Diffusion Models 
- [NeurIPS] Q-DM: An Efficient Low-bit Quantized Diffusion Model
- [NeurIPS] PTQD: Accurate Post-Training Quantization for Diffusion Models 
- [NeurIPS] Temporal Dynamic Quantization for Diffusion Models
- [ArXiv] Towards Accurate Data-free Quantization for Diffusion Models
- [ArXiv] Finite Scalar Quantization: VQ-VAE Made Simple 
- [ArXiv] EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models
- QuEST: Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

## Other

- Accelerating Parallel Sampling of Diffusion Models

### Notes

- This repository is a work in progress, and contributions are welcome. Please feel free to submit pull requests or open issues for any additions, corrections, or suggestions.
- The code links provided are for reference purposes only. Please refer to the respective papers or repositories for more details and instructions on using the code.
