# Connector-S

> 🔥 **A collection of must-read papers and resources related to connectors in MLLMs.**
>
> The organization of papers refers to our survey [**"Connector-S: A Survey of Connectors in Multi-modal Large Language Models"**](https://arxiv.org/abs/2502.11453). [![Paper page](https://huggingface.co/datasets/huggingface/badges/raw/main/paper-page-sm-dark.svg)](https://arxiv.org/abs/2502.11453)
>
> Please let us know if you find out a mistake or have any suggestions.
>
> If you find our survey useful for your research, please cite the following paper:
```
@article{zhu2025connector,
  title={Connector-S: A Survey of Connectors in Multi-modal Large Language Models},
  author={Zhu, Xun and Zhang, Zheng and Chen, Xi and Shi, Yiming and Li, Miao and Wu, Ji},
  journal={arXiv preprint arXiv:2502.11453},
  year={2025}
}
```

## 🔔 News

- ✨ [2025/03/16] We create this repository to maintain and expand a paper list on connectors in MLLMs. More papers are coming soon!

- 💥 [2025/02/18] Our survey is released! See [Connector-S](https://arxiv.org/abs/2502.11453) for the paper!


## 🌟 Introduction

With the rapid advancements in multi-modal large language models (MLLMs), connectors play a pivotal role in bridging diverse modalities and enhancing model performance. However, the design and evolution of connectors have not been comprehensively analyzed, leaving gaps in understanding how these components function and hindering the development of more powerful connectors.

we systematically review the current progress of connectors in MLLMs and present a structured taxonomy that categorizes connectors into atomic operations (mapping, compression, mixture of experts) and holistic designs (multi-layer, multi-encoder, multi-modal scenarios), highlighting their technical contributions and advancements. Furthermore, we list several promising research frontiers and challenges, including high-resolution input, dynamic compression, guide information selection, combination strategy, and interpretability.


## Table of Content

- [Connector-S](#connector-s)
  - [🔔 News](#-news)
  - [🌟 Introduction](#-introduction)
  - [Table of Content](#table-of-content)
  - [Paper List](#paper-list)
    - [Atomic Connector Operations](#atomic-connector-operations)
      - [Mapping](#mapping)
      - [Compression](#compression)
      - [Mixture of Experts](#mixture-of-experts)
    - [Holistic Connector Designs](#holistic-connector-designs)
      - [Multi-Layer Scenario](#multi-layer-scenario)
      - [Multi-Encoder Scenario](#multi-encoder-scenario)
      - [Multi-Modal Scenario](#multi-modal-scenario)
    - [Future Directions and Challenges](#future-directions-and-challenges)  
      - [High-Resolution Input](#high-resolution-input)  
      - [Dynamic Compression](#dynamic-compression)  
      - [Guide Information Selection](#guide-information-selection)  
      - [Combination Strategy](#combination-strategy)  
      - [Interpretability](#interpretability)
## Paper List

### Atomic Connector Operations

Atomic connector operations refer to the basic components of MLLM connectors, which are designed as simple yet versatile units tailored to different functional requirements of basic scenarios.
By utilizing these atomic operations, connectors can achieve mapping, compression, and expert integration.
Furthermore, they can be combined to create more complex connectors, bridging the modality gap in a targeted and flexible way. 

#### Mapping

Mapping operations first flatten 2D or 3D features into 1D in a specific order and directly align the dimension of representations from other modalities with textual token embeddings.

##### Linear
- [NeurIPS 23] **"Visual Instruction Tuning"**. *Liu et al.* [[Paper](https://arxiv.org/pdf/2304.08485)] [[Resource](https://llava-vl.github.io)]
- [arXiv 23] **"LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models"**. *Zhang et al.* [[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05918.pdf)] [[Resource](https://github.com/UX-Decoder/LLaVA-Grounding)]
- [NeurIPS 24] **"Lumen: Unleashing Versatile Vision-Centric Capabilities of Large Multimodal Models"**. *Jiao et al.* [[Paper](https://arxiv.org/pdf/2403.07304)]
- [NeurIPS 24] **"VITRON: A Unified Pixel-level Vision LLM for Understanding, Generating, Segmenting, Editing"**. *Fei et al.* [[Paper](https://openreview.net/pdf?id=kPmSfhCM5s)] [[Resource](https://github.com/SkyworkAI/Vitron)]
- [EMNLP 24] **"M2PT: Multimodal Prompt Tuning for Zero-shot Instruction Learning"**. *Wang et al.* [[Paper](https://arxiv.org/pdf/2409.15657)] [[Resource](https://github.com/William-wAng618/M2PT)] 
- [CIKM 24] **"ChefFusion: Multimodal Foundation Model Integrating Recipe and Food Image Generation"**. *Li et al.* [[Paper](https://arxiv.org/pdf/2409.12010v1)] [[Resource](https://github.com/Peiyu-Georgia-Li/ChefFusion-Multimodal-Foundation-Model-Integrating-Recipe-and-Food-Image-Generation)]
- [CVPR 24] **"Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs"**. *Ranasinghe et al.* [[Paper](https://arxiv.org/pdf/2404.07449)]
- [CVPR 24] **"VTimeLLM: Empower LLM to Grasp Video Moments"**. *Huang et al.* [[Paper](https://arxiv.org/pdf/2311.18445)] [[Resource](https://github.com/huangb23/VTimeLLM)]
- [CVPR 24] **"LocLLM: Exploiting Generalizable Human Keypoint Localization via Large Language Model"**. *Wang et al.* [[Paper](https://arxiv.org/pdf/2406.04659)] [[Resource](https://github.com/kennethwdk/LocLLM)]
- [ACM MM 24]**"LLaVA-VSD: Large Language-and-Vision Assistant for Visual Spatial Description"**. *Jin et al.* [[Paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3688992)] [[Resource](https://github.com/swordlidev/LLaVA-VSD)]
- [Computers and Education: Artificial Intelligence] **"LLaVA-docent: Instruction tuning with multimodal large language model to support art appreciation education"**. *Lee et al.* [[Paper](https://arxiv.org/pdf/2402.06264)]
- [AAAI 24] **"BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions"**. *Hu et al.* [[Paper](https://arxiv.org/abs/2308.09936)] [[Resource](https://github.com/mlpc-ucsd/BLIVA)]
- [arXiv 24] **"LM4LV: A Frozen Large Language Model for Low-level Vision Tasks"**. *Zheng et al.* [[Paper](https://arxiv.org/pdf/2405.15734v2)] [[Resource](https://github.com/bytetriper/LM4LV)]
- [ICLR 25] **"mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models"**. *Ye et al.* [[Paper](https://arxiv.org/pdf/2408.04840)] [[Resource](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-240728)]

##### MLP
- [ECCV 24] **"ShareGPT4V: Improving Large Multi-Modal Models with Better Captions"**. *Chen et al.* [[Paper](https://arxiv.org/pdf/2311.12793)] [[Resource](https://sharegpt4v.github.io/)]
- [ACL 24] **"GeoGPT4V:Towards Geometric Multi-modal Large Language Models with Geometric Image Generation"**. *Cai et al.* [[Paper](https://arxiv.org/pdf/2406.11503)] [[Resource](https://github.com/Lanyu0303/GeoGPT4V_Project)]
- [Science China Information Sciences 24] **"How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites"**. *Chen et al.* [[Paper](https://arxiv.org/pdf/2404.16821)] [[Resource](https://github.com/OpenGVLab/InternVL)]
- [CVPR 24] **"CogVLM: Visual Expert for Pretrained Language Models"**. *Wang et al.* [[Paper](https://arxiv.org/abs/2311.03079)] [[Resource](https://github.com/THUDM/CogVLM)]
- [CVPR 24] **"Multi-modal Instruction Tuned LLMs with Fine-grained Visual Perception"**. *He et al.* [[Paper](https://arxiv.org/pdf/2403.02969)] [[Resource](https://github.com/jwh97nn/AnyRef)]
- [CVPR 24] **"Improved Baselines with Visual Instruction Tuning"**. *Liu et al.* [[Paper](https://arxiv.org/pdf/2310.03744)] [[Resource](https://github.com/haotian-liu/LLaVA)] 
- [EMNLP 24] **"MMNeuron: Discovering Neuron-Level Domain-Specific Interpretation in Multimodal Large Language Model"**. *Huo et al.* [[Paper](https://aclanthology.org/2024.emnlp-main.387.pdf)] [[Resource](https://github.com/Z1zs/MMNeuron)]
- [EMNLP 24] **"Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models"**. *Jiang et al.* [[Paper](https://aclanthology.org/2024.finding)] [[Resource](https://github.com/jiangsongtao/Med-MoE)]
- [EMNLP 24] **"Self-Bootstrapped Visual-Language Model for Knowledge Selection and Question Answering"**. *Hao et al.* [[Paper](https://aclanthology.org/2024.emnlp-main.110.pdf)] [[Resource](https://github.com/haodongze/Self-KSel-QAns)]
- [NeurIPS 24] **"ControlMLLM: Training-Free Visual Prompt Learning for Multimodal Large Language Models"**. *Wu et al.* [[Paper](https://arxiv.org/pdf/2407.21534)] [[Resource](https://github.com/mrwu-mac/ControlMLLM)]
- [ICLR 24] **"DREAMLLM: SYNERGISTIC MULTIMODAL COMPREHENSION AND CREATION"**. *Dong et al.* [[Paper](https://arxiv.org/pdf/2309.11499)] [[Resource](https://dreamllm.github.io/)]
- [ICME 24] **"3DMIT: 3D MULTI-MODAL INSTRUCTION TUNING FOR SCENE UNDERSTANDING"**. *Li et al.* [[Paper](https://arxiv.org/pdf/2401.03201)] [[Resource](https://github.com/staymylove/3DMIT)]
- [arXiv 23] **"VCoder: Versatile Vision Encoders for Multimodal Large Language Models"**. *Jain et al.* [[Paper](https://arxiv.org/pdf/2312.14233)] [[Resource](https://github.com/SHI-Labs/VCoder)]
- [arXiv 24] **"ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models"**. *Ge et al.* [[Paper](https://arxiv.org/pdf/2405.15738)] [[Resource](https://github.com/alibaba/conv-llava)]
- [arXiv 24] **"InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks"**. *Chen et al.* [[Paper](https://arxiv.org/pdf/2312.14238)] [[Resource](https://github.com/OpenGVLab/InternVL)]
- [arXiv 24] **"Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs"**. *Ranasinghe et al.* [[Paper](https://arxiv.org/pdf/2404.07449)]
- [arXiv 24] **"Yi: Open Foundation Models by 01.AI"**. *Young et al.* [[Paper](https://arxiv.org/pdf/2403.04652)] [[Resource](https://github.com/01-ai/Yi)]  
- [arXiv 24] **"MoE-LLaVA: Mixture of Experts for Large Vision-Language Models"**. *Lin et al.* [[Paper](https://arxiv.org/pdf/2401.15947)] [[Resource](https://github.com/PKU-YuanGroup/MoE-LLaVA)]
- [arXiv 24] **"RoboMP2: A Robotic Multimodal Perception-Planning Framework with Multimodal Large Language Models"**. *Lv et al.* [[Paper](https://arxiv.org/pdf/2404.04929)] [[Resource](https://aopolin-lv.github.io/RoboMP2.github.io/)]
#### Compression

##### Spatial Relation

###### Simple Operation 
- [NeurIPS 24] **"DeTikZify: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ"**. *belouadi et al.* [[Paper](https://arxiv.org/pdf/2405.15306)] [[Resource](https://github.com/potamides/DeTikZify)]
- [CVPR 24] **"Generative Multimodal Models are In-Context Learners"**. *Sun et al.* [[Paper](https://arxiv.org/pdf/2312.13286)] [[Resource](https://github.com/baaivision/Emu)]
- [ACM MM 24] **"EAGLE: Egocentric AGgregated Language-video Engine"**. *Bi et al.* [[Paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3681618)] [[Resource](https://github.com/yaolinli/DeCo)]
- [ACL 24] **"Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models"**. *Yao et al.* [[Paper](https://aclanthology.org/2024.acl-long.679.pdf)] [[Resource](https://github.com/mbzuai-oryx/Video-ChatGPT)]
- [ICLR 25] **"PLLaVA : Parameter-free LLaVA Extension from Images to Videos for Video Dense Captioning"**. *Xu et al.* [[Paper](https://arxiv.org/pdf/2404.16994)] [[Resource](https://github.com/magic-research/PLLaVA)]
- [arXiv 23] **"MiniGPT-v2: Large Language Model As a Unified Interface for Vision-Language Multi-task Learning"**. *Chen et al.* [[Paper](https://arxiv.org/pdf/2310.09478)] [[Resource](https://github.com/Vision-CAIR/MiniGPT-4)]
- [arXiv 24] **"Accelerating Pre-training of Multimodal LLMs via Chain-of-Sight"**. *Huang et al.* [[Paper](https://arxiv.org/pdf/2407.15819)] [[Resource](https://chain-of-sight.github.io/)]
- [arXiv 24] **"DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models"**. *Yao et al.* [[Paper](https://arxiv.org/pdf/2405.20985)] [[Resource](https://github.com/yaolinli/DeCo)]
###### CNN
- [arXiv 23] **"MACAW-LLM: MULTI-MODAL LANGUAGE MODELING WITH IMAGE, AUDIO, VIDEO, AND TEXT INTEGRATION"**. *Lyu et al.* [[Paper](https://arxiv.org/pdf/2306.09093)] [[Resource](https://github.com/lyuchenyang/Macaw-LLM)]
- [CVPR 24] **"Honeybee: Locality-enhanced Projector for Multimodal LLM"**. *Cha et al.* [[Paper](https://arxiv.org/pdf/2312.06742)] [[Resource](https://github.com/kakaobrain/honeybee)]  
- [ECCV 24] **"MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training"**. *McKinzie et al.* [[Paper](https://arxiv.org/pdf/2403.09611)]
###### Variants 
- [CVPR 24] **"Honeybee: Locality-enhanced Projector for Multimodal LLM"**. *Cha et al.* [[Paper](https://arxiv.org/pdf/2312.06742)] [[Resource](https://github.com/kakaobrain/honeybee)]   
- [NeurIPS 2024] **"MoME: Mixture of Multimodal Experts for Generalist Multimodal Large Language Models"**. *Shen et al.* [[Paper](https://arxiv.org/pdf/2407.12709)] [[Resource](https://github.com/JiuTian-VL/MoME)] 
##### Semantic Perception

###### Q-Former 
- [NeurIPS 23] **"InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning"**. *Dai et al.* [[Paper](https://arxiv.org/pdf/2305.06500)] [[Resource](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)]
- [ICLR 23] **"BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"**. *Li et al.* [[Paper](https://arxiv.org/pdf/2301.12597)] [[Resource](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)]
- [ACL ARR 24] **"UniFashion: A Unified Vision-Language Model for Multimodal Fashion Retrieval and Generation"**. *Zhao et al.* [[Paper](https://arxiv.org/pdf/2408.11305)] [[Resource](https://github.com/xiangyu-mm/UniFashion)]
- [CVPR 24] **"SNIFFER: Multimodal Large Language Model for Explainable Out-of-Context Misinformation Detection"**. *Qi et al.* [[Paper](https://arxiv.org/pdf/2403.03170)] [[Resource](https://pengqi.site/Sniffer)]
- [CVPR 24] **"Visual Delta Generator with Large Multi-modal Models for Semi-supervised Composed Image Retrieval"**. *Jang et al.* [[Paper](https://arxiv.org/pdf/2404.15516)] [[Resource](https://github.com/youngkyunJang/VDG)
- [CVPR 24] **"Embodied Multi-Modal Agent trained by an LLM from a Parallel TextWorld"**. *Yang et al.* [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Embodied_Multi-Modal_Agent_trained_by_an_LLM_from_a_Parallel_CVPR_2024_paper.pdf)] [[Resource](https://github.com/stevenyangyj/Emma-Alfworld)]
- [ACM TKDD 24] **"TOMGPT: Reliable Text-Only Training Approach for Cost-Effective Multi-modal Large Language Model"**. *Chen et al.* [[Paper](https://dl.acm.org/doi/pdf/10.1145/3654674)]
- [ICLR 24] **"MMICL: EMPOWERING VISION-LANGUAGE MODEL WITH MULTI-MODAL IN-CONTEXT LEARNING"**. *Zhao et al.* [[Paper](https://arxiv.org/pdf/2309.07915)] [[Resource](https://github.com/PKUnlp-icler/MIC)]
- [ICLR 24] **"MINIGPT-4:ENHANCING VISION-LANGUAGE UNDERSTANDING WITH ADVANCED LARGE LANGUAGE MODELS"**. *Zhu et al.* [[Paper](https://openreview.net/pdf?id=1tZbq88f27)] [[Resource](https://minigpt-4.github.io)]
- [ICLR 24] **"EMU: GENERATIVE PRETRAINING IN MULTIMODALITY"**. *Sun et al.* [[Paper](https://arxiv.org/pdf/2307.05222)] [[Resource](https://github.com/baaivision/Emu)]
- [ICML 24] **"TinyGPT-V: Efficient Multimodal Large Language Model via Small Backbones"**. *Yuan et al.* [[Paper](https://arxiv.org/pdf/2312.16862)]
- [arXiv 24] **"Towards Event-oriented Long Video Understanding"**. *Du et al.* [[Paper](https://openreview.net/attachment?id=6YjJklAAQ9&name=pdf)] [[Resource](https://github.com/RUCAIBox/Event-Bench)]
- [ACL 24] **"UNIMO-G: Unified Image Generation through Multimodal Conditional Diffusion"**. *Li et al.* [[Paper](https://aclanthology.org/2024.acl-long.335.pdf)] [[Resource](https://unimo-ptm.github.io/)]
- [ACM MM 24] **"Hypergraph Multi-modal Large Language Model: Exploiting EEG and Eye-tracking Modalities to Evaluate Heterogeneous Responses for Video Understanding"**. *Wu et al.* [[Paper](https://openreview.net/attachment?id=mFy8n4Gdc9&name=pdf)] [[Resource](https://github.com/mininglamp-MLLM/HMLLM)]
- [ACM MM 24] **"CREAM: Coarse-to-Fine Retrieval and Multi-modal Efficient Tuning for Document VQA"**. *Zhang et al.* [[Paper](https://openreview.net/attachment?id=uxxdE9HFGI&name=pdf)]
- [ACM MM 24] **"GPT4Video: A Unified Multimodal Large Language Model for Instruction-Followed Understanding and Safety-Aware"**. *Wang et al.* [[Paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3681464)]
- [MICCAI 24] **"PathAlign: A vision–language model for whole slide images in histopathology"**. *Ahmed et al.* [[Paper](https://openreview.net/attachment?id=Q2nSLk9hKT&name=pdf)]
- [BIBM 24] **"C2RG: Parameter-efficient Adaptation of 3D Vision and Language Foundation Model for Coronary CTA Report Generation"**. *Ye et al.* [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10821939)]
- [NeurIPS 24] **"What matters when building vision-language models?"**. *Laurençon et al.* [[Paper](https://arxiv.org/pdf/2405.02246)] [[Resource](https://huggingface.co/collections/HuggingFaceM4/idefics2-661d1971b7c50831dd3ce0fe)]
- [AAAI 24] **"Structure-aware multimodal sequential learning for visual dialog"**. *Kim et al.* [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29219)]
- [AAAI 24] **"BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions"**. *Hu et al.* [[Paper](https://arxiv.org/abs/2308.09936)] [[Resource](https://github.com/mlpc-ucsd/BLIVA)]
- [AAAI 25] **"PlanLLM: Video Procedure Planning with Refinable Large Language Models"**. *Yang et al.* [[Paper](https://www.arxiv.org/pdf/2412.19139)] [[Resource](https://github.com/idejie/PlanLLM)]
###### Resampler
- [NeurIPS 22] **"Flamingo: a Visual Language Model for Few-Shot Learning"**. *Alayrac et al.* [[Paper](https://arxiv.org/pdf/2204.14198)]
- [NeurIPS 24] **"Voila-A: Aligning Vision-Language Models with User's Gaze Attention"**. *Yan et al.* [[Paper](https://bytez.com/docs/neurips/94630/paper)] 
- [CVPR 24] **"Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models"**. *Li et al.* [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Monkey_Image_Resolution_and_Text_Label_Are_Important_Things_for_CVPR_2024_paper.pdf)] [[Resource](https://github.com/Yuliang-Liu/Monkey)]
- [ACL 24] **"InfiMM: Advancing Multimodal Understanding with an Open-Sourced Visual Language Model"**. *Liu et al.* [[Paper](https://aclanthology.org/2024.findings-acl.27.pdf)] [[Resource](https://huggingface.co/Infi-MM)]
###### Abstractor
- [ACM MM 24] **"mPLUG-PaperOwl: Scientific Diagram Analysis with the Multimodal Large Language Model"**. *Hu et al.* [[Paper](https://arxiv.org/pdf/2311.18248)] [[Resource](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/PaperOwl)]
- [CVPR 24] **"mPLUG-OwI2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration"**. *Ye et al.* [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10657415)] [[Resource](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)]
- [ICML 24] **"GeoReasoner: Geo-localization with Reasoning in Street Views using a Large Vision-Language Model"**. *Li et al.* [[Paper](https://arxiv.org/pdf/2406.18572)] [[Resource](https://github.com/lingli1996/GeoReasoner)]
- [arXiv 24] **"Adaptive Image Quality Assessment via Teaching Large Multimodal Model to Compare"**. *Zhu et al.* [[Paper](https://arxiv.org/pdf/2405.19298)] [[Resource](https://compare2score.github.io/)]
- [arXiv 24] **"Q-ALIGN: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels"**. *Wu et al.* [[Paper](https://arxiv.org/pdf/2312.17090)] [[Resource](https://github.com/Q-Future/Q-Align)]
###### Variants 
- [EMNLP 24] **"Query-based Cross-Modal Projector Bolstering Mamba Multimodal LLM"**. *Eom et al.* [[Paper](https://aclanthology.org/2024.findings-emnlp.827.pdf)]
- [arXiv 24] **"ParGo: Bridging Vision-Language with Partial and Global Views"**. *Wang et al.* [[Paper](https://arxiv.org/pdf/2408.12928)] [[Resource](https://github.com/bytedance/ParGo)] 
#### Mixture of Experts

##### Vanilla MoE
- [NeurIPS 2024] **"CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts"**. *Cha et al.* [[Paper](https://arxiv.org/pdf/2405.05949)] [[Resource](https://github.com/SHI-Labs/CuMo)]
- [BIBM 2024]**"SurgFC: Multimodal Surgical Function Calling Framework on the Demand of Surgeons"**. *Chen et al.* [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10821976)] [[Resource](https://github.com/franciszchen/SurgFC)]
- [ICLR 2025] **"CHARTMOE: MIXTURE OF DIVERSELY ALIGNED EXPERT CONNECTOR FOR CHART UNDERSTANDING"**. *Xu et al.* [[Paper](https://arxiv.org/pdf/2409.03277)] [[Resource](https://huggingface.co/IDEA-FinAI/chartmoe)]  
##### X-Guided MoE

###### Modality-Guided 
- [CVPR 2024] **"OneLLM: One Framework to Align All Modalities with Language"**. *Han et al.* [[Paper](https://arxiv.org/pdf/2312.03700)] [[Resource](https://github.com/csuhan/OneLLM)] 
###### Text-Guided
- [ACM MM 2024] **"Q-MoE: Connector for MLLMs with Text-Driven Routing"**. *Wang et al.* [[Paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3681369)]
###### Task-Guided
- [NeurIPS 2024] **"Uni-Med: A Unified Medical Generalist Foundation Model For Multi-Task Learning Via Connector-MoE"**. *Zhu et al.* [[Paper](https://arxiv.org/abs/2409.17508)] [[Resource](https://github.com/tsinghua-msiip/Uni-Med)]  
##### Variant MoE
- [CVPR 24] **"V∗: Guided Visual Search as a Core Mechanism in Multimodal LLMs"**. *Wu and Xie* [[Paper](https://arxiv.org/pdf/2312.14135)] [[Resource](https://github.com/penghao-wu/vstar)]
### Holistic Connector Designs

##### Multi-Layer Scenario
- [ECCV 24] **"Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models"**. *Ma et al.* [[Paper](https://arxiv.org/pdf/2404.13013)] [[Resource](https://groma-mllm.github.io/)]
- [CVPR 24] **"GLaMM: Pixel Grounding Large Multimodal Model"**. *Rasheed et al.* [[Paper](https://arxiv.org/pdf/2311.03356)] [[Resource](https://github.com/mbzuai-oryx/groundingLMM)]
- [CVPR 24] **"LION : Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge"**. *Chen et al.* [[Paper](https://arxiv.org/pdf/2311.11860)] [[Resource](https://github.com/rshaojimmy/JiuTian)] 
- [NeurIPS 24] **"Dense Connector for MLLMs"**. *Yao et al.* [[Paper](https://arxiv.org/pdf/2405.13800)] [[Resource](https://github.com/HJYao00/DenseConnector)]  
- [arXiv 24] **"TokenPacker:Efficient Visual Projector for Multimodal LLM"**. *Li et al.* [[Paper](https://arxiv.org/pdf/2407.02392)] [[Resource](https://github.com/CircleRadon/TokenPacker)]  
- [arXiv 24] **"MMFuser: Multimodal Multi-Layer Feature Fuser for Fine-Grained Vision-Language Understanding"**. *Cao et al.* [[Paper](https://arxiv.org/pdf/2410.11829)] [[Resource](https://github.com/yuecao0119/MMFuser)]
- [arXiv 24] **"TextHawk: Exploring Efficient Fine-Grained Perception of Multimodal Large Language Models"**. *Yu et al.* [[Paper](https://arxiv.org/pdf/2404.09204)] [[Resource](https://github.com/yuyq96/TextHawk)]
##### Multi-Encoder Scenario
- [arXiv 24] **"DeepSeek-VL: Towards Real-World Vision-Language Understanding"**. *Lu et al.* [[Paper](https://arxiv.org/pdf/2403.05525)] [[Resource](https://github.com/deepseek-ai/DeepSeek-VL)]
- [arXiv 24] **"SPHINX: THE JOINT MIXING OF WEIGHTS, TASKS,AND VISUAL EMBEDDINGS FOR MULTI-MODAL LARGE LANGUAGE MODELS"**. *Lin et al.* [[Paper](https://arxiv.org/pdf/2311.07575)] [[Resource](https://github.com/Alpha-VLLM/LLaMA2-Accessory)]
- [ICML 24] **"SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models"**. *Liu et al.* [[Paper](https://openreview.net/pdf?id=tDMlQkJRhZ)] [[Resource](https://github.com/Alpha-VLLM/LLaMA2-Accessory)]
- [CoRL 24] **"OpenVLA:An Open-Source Vision-Language-Action Model"**. *Kim et al.* [[Paper](https://openreview.net/attachment?id=ZMnD6QZAE6&name=pdf)] [[Resource](https://openvla.github.io/)]
- [ICLR 24] **"From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models"**. *Jiang et al.* [[Paper](https://arxiv.org/pdf/2310.08825)]
- [CVPR 24] **"Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs"**. *Tong et al.* [[Paper](https://arxiv.org/pdf/2401.06209)]
- [ECCV 24] **"BRAVE : Broadening the visual encoding of vision-language models"**. *Kar et al.* [[Paper](https://arxiv.org/pdf/2404.07204)] [[Resource](https://github.com/kyegomez/BRAVE-ViT-Swarm)]
- [ACM MM 24] **"LLaVA-Ultra: Large Chinese Language and Vision Assistant for Ultrasound"**. *Guo et al.* [[Paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3681584)]
- [NeurIPS 24] **"MoME: Mixture of Multimodal Experts for Generalist Multimodal Large Language Models"**. *Shen et al.* [[Paper](https://arxiv.org/pdf/2407.12709)] [[Resource](https://github.com/JiuTian-VL/MoME)]   
- [NeurIPS 24] **"Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs"**. *Tong et al.* [[Paper](https://arxiv.org/pdf/2406.16860)] [[Resource](https://github.com/cambrian-mllm/cambrian)]  
- [NeurIPS 24] **"MaVEn: An Effective Multi-granularity Hybrid Visual Encoding Framework for Multimodal Large Language Model"**. *Jiang et al.* [[Paper](https://bytez.com/docs/neurips/94216/paper)]
- [ICLR 25] **"EAGLE: EXPLORING THE DESIGN SPACE FOR MULTIMODAL LLMS WITH MIXTURE OF ENCODERS"**. *Shi et al.* [[Paper](https://openreview.net/pdf?id=Y2RW9EVwhT)] [[Resource](https://github.com/NVlabs/Eagle)]  
- [AAAI 25] **"Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference"**. *Zhao et al.* [[Paper](https://arxiv.org/pdf/2403.14520)] [[Resource](https://sites.google.com/view/cobravlm)]
##### Multi-Modal Scenario
- [TMLR 24] **"LLaVA-OneVision: Easy Visual Task Transfer"**. *Li et al.* [[Paper](https://arxiv.org/pdf/2408.03326)] [[Resource](https://llava-vl.github.io/blog/2024-08-05-llava-onevision)]
- [CVPR 24] **"OneLLM: One Framework to Align All Modalities with Language"**. *Han et al.* [[Paper](https://arxiv.org/pdf/2312.03700)] [[Resource](https://github.com/csuhan/OneLLM)]
- [ACL 24] **"Recognizing Everything from All Modalities at Once:Grounded Multimodal Universal Information Extraction"**. *Zhang et al.* [[Paper](https://arxiv.org/pdf/2305.16355)] [[Resource](https://haofei.vip/MUIE)]
- [ACL 24] **"GroundingGPT: Language Enhanced Multi-modal Grounding Model"**. *Li et al.* [[Paper](https://aclanthology.org/2024.acl-long.360.pdf)] [[Resource](https://github.com/lzw-lzw/GroundingGPT)]
- [ICML 24] **"MACAW-LLM: MULTI-MODAL LANGUAGE MODELING WITH IMAGE, AUDIO, VIDEO, AND TEXT INTEGRATION"**. *Lyu et al.* [[Paper](https://arxiv.org/pdf/2306.09093)] [[Resource](https://github.com/lyuchenyang/Macaw-LLM)]
- [ECCV 24] **"Meerkat: Audio-Visual Large Language Model for Grounding in Space and Time"**. *Chowdhury et al.* [[Paper](https://arxiv.org/pdf/2407.01851)] [[Resource](https://github.com/schowdhury671/meerkat)]    
- [ECCV 25] **"CAT : Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios"**. *Ye et al.* [[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01541.pdf)] [[Resource](https://github.com/rikeilong/Bay-CAT)]  
- [EMNLP 24] **"AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model"**. *Moon et al.* [[Paper](https://arxiv.org/pdf/2309.16058)]
- [arXiv 23] **"VideoPoet: A Large Language Model for Zero-Shot Video Generation"**. *Kondratyuk et al.* [[Paper](https://arxiv.org/pdf/2312.14125)] [[Resource](https://sites.research.google/videopoet/)]
- [arXiv 23] **"PandaGPT:One Model To Instruction-Follow Them All"**. *Su et al.* [[Paper](https://arxiv.org/pdf/2305.16355)] [[Resource](https://panda-gpt.github.io/)]
- [arXiv 24] **"Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution"**. *Wang et al.* [[Paper](https://arxiv.org/pdf/2409.12191)] [[Resource](https://github.com/QwenLM/Qwen2-VL)]
- [arXiv 24] **"World Knowledge-Enhanced Reasoning Using Instruction-guided Interactor in Autonomous Driving"**. *Zhai et al.* [[Paper](https://arxiv.org/pdf/2412.06324)]
### Future Directions and Challenges

##### High-Resolution Input
- [EMNLP 23] **"UReader: Universal OCR-free Visually-situated Language Understanding with Multimodal Large Language Model"**. *Ye et al.* [[Paper](https://aclanthology.org/2023.findings-emnlp.187.pdf)]
- [Science China Information Sciences 24] **"How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites"**. *Chen et al.* [[Paper](https://arxiv.org/pdf/2404.16821)] [[Resource](https://github.com/OpenGVLab/InternVL)]  
- [arXiv 24] **"SPHINX: THE JOINT MIXING OF WEIGHTS, TASKS,AND VISUAL EMBEDDINGS FOR MULTI-MODAL LARGE LANGUAGE MODELS"**. *Lin et al.* [[Paper](https://arxiv.org/pdf/2311.07575)] [[Resource](https://github.com/Alpha-VLLM/LLaMA2-Accessory)]
- [ICML 24] **"SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models"**. *Liu et al.* [[Paper](https://openreview.net/pdf?id=tDMlQkJRhZ)] [[Resource](https://github.com/Alpha-VLLM/LLaMA2-Accessory)]
- [BIBM 24] **"PA-LLaVA: A Large Language-Vision Assistant for Human Pathology Image Understanding"**. *Dai et al.* [[Paper](https://arxiv.org/pdf/2408.09530)] [[Resource](https://github.com/ddw2AIGROUP2CQUPT/PA-LLaVA)]
- [ECCV 24] **"LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images"**. *Xu et al.* [[Paper](https://arxiv.org/pdf/2403.11703)] [[Resource](https://github.com/thunlp/LLaVA-UHD)]
- [CVPR 24] **"Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models"**. *Li et al.* [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Monkey_Image_Resolution_and_Text_Label_Are_Important_Things_for_CVPR_2024_paper.pdf)] [[Resource](https://github.com/Yuliang-Liu/Monkey)]
- [NeurIPS 24] **"VisionLLM v2: An End-to-End Generalist Multimodal Large Language Model for Hundreds of Vision-Language Tasks"**. *Wu et al.* [[Paper](https://nips.cc/virtual/2024/poster/93655)] [[Resource](https://github.com/OpenGVLab/VisionLLM)]
- [NeurIPS 24] **"DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs"**. *Meng et al.* [[Paper](https://nips.cc/virtual/2024/poster/94201)] [[Resource](https://deepstack-vl.github.io/)]
- [(NeurIPS 24] **"InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD"**. *Dong et al.* [[Paper](https://openreview.net/pdf/1155d659fc23c66face1a69fc64ca5781c59f825.pdf)] [[Resource](https://github.com/InternLM/InternLM-XComposer)]
- [AAAI 25] **"HiRED: Attention-Guided Token Dropping for Efficient Inference of High-Resolution Vision-Language Models"**. *Arif et al.* [[Paper](https://arxiv.org/pdf/2408.10945)] [[Resource](https://github.com/hasanar1f/HiRED)] 
##### Dynamic Compression
- [arXiv 24] **"FocusLLaVA: A Coarse-to-Fine Approach for Efficient and Effective Visual Token Compression"**. *Zhu et al.* [[Paper](https://arxiv.org/pdf/2411.14228)]
- [NeurIPS 24] **"Visual Anchors Are Strong Information Aggregators For Multimodal Large Language Model"**. *Liu et al.* [[Paper](https://arxiv.org/pdf/2405.17815)] [[Resource](https://github.com/liuhaogeng/Anchor-Former)]
- [ICLR 24] **"UNIFIED LANGUAGE-VISION PRETRAINING IN LLM WITH DYNAMIC DISCRETE VISUAL TOKENIZATION"**. *Jin et al.* [[Paper](https://arxiv.org/pdf/2309.04669)] [[Resource](https://github.com/jy0205/LaVIT)]
- [AAAI 25] **"DocKylin: A Large Multimodal Model for Visual Document Understanding with Efficient Visual Slimming"**. *Zhang et al.* [[Paper](https://arxiv.org/pdf/2406.19101)]

##### Guide Information Selection
- [NeurIPS 24] **"MaVEn: An Effective Multi-granularity Hybrid Visual Encoding Framework for Multimodal Large Language Model"**. *Jiang et al.* [[Paper](https://bytez.com/docs/neurips/94216/paper)]
- [ACM MM 24] **"Semantic Alignment for Multimodal Large Language Models"**. *Wu et al.* [[Paper](https://arxiv.org/pdf/2412.06324)] [[Resource](https://mccartney01.github.io/SAM)]
- [ACL 24] **"MLeVLM: Improve Multi-level Progressive Capabilities based on Multimodal Large Language Model for Medical Visual Question Answering"**. *Xu et al.* [[Paper](https://aclanthology.org/2024.findings-acl.296.pdf)] [[Resource](https://github.com/RyannChenOO/MLeVLM)]
- [AAAI 25] **"Towards a Multimodal Large Language Model with Pixel-Level Insight for Biomedicine"**. *Huang et al.* [[Paper](https://arxiv.org/pdf/2412.09278)] [[Resource](https://github.com/ShawnHuang497/MedPLIB)]
- [arXiv 24] **"TG-LLaVA: Text Guided LLaVA via Learnable Latent Embeddings"**. *Yan et al.* [[Paper](https://arxiv.org/abs/2409.09564)]
- [arXiv 24] **"PPLLAVA: VARIED VIDEO SEQUENCE UNDERSTANDING WITH PROMPT GUIDANCE"**. *Liu et al.* [[Paper](https://arxiv.org/pdf/2411.02327)] [[Resource](https://github.com/farewellthree/PPLLaVA)]  
- [arXiv 24] **"World Knowledge-Enhanced Reasoning Using Instruction-guided Interactor in Autonomous Driving"**. *Zhai et al.* [[Paper](https://arxiv.org/pdf/2412.06324)]

##### Combination Strategy
- [NeurIPS 24] **"Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs"**. *Tong et al.* [[Paper](https://arxiv.org/pdf/2406.16860)] [[Resource](https://github.com/cambrian-mllm/cambrian)]
- [ICLR 25] **"EAGLE: EXPLORING THE DESIGN SPACE FOR MULTIMODAL LLMS WITH MIXTURE OF ENCODERS"**. *Shi et al.* [[Paper](https://openreview.net/pdf?id=Y2RW9EVwhT)] [[Resource](https://github.com/NVlabs/Eagle)]

##### Interpretability
- [EMNLP 24] **"MMNeuron: Discovering Neuron-Level Domain-Specific Interpretation in Multimodal Large Language Model"**. *Huo et al.* [[Paper](https://aclanthology.org/2024.emnlp-main.387.pdf)] [[Resource](https://github.com/Z1zs/MMNeuron)]  
- [arXiv 24] **"DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models"**. *Yao et al.* [[Paper](https://arxiv.org/pdf/2405.20985)] [[Resource](https://github.com/yaolinli/DeCo)]
