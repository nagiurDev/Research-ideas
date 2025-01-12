# LLM Research Ideas

## `Lecture - 13`: `LLM - Deployment`

**Quantization and Compression:**

- **Novel Activation Quantization Techniques:** The lecture highlights the limitations of existing quantization methods. `A research project could investigate a new activation quantization method that addresses the issues raised, like the problem of outliers in activations. This could involve developing a new algorithm or adapting an existing one for use with LLMs, focusing on improving accuracy and/or minimizing the quantization error`.

- **Mixed-Precision Quantization for LLMs:** The slides discuss the concept of keeping only 1% of salient weights unquantized. A thesis could explore a new strategy for mixed-precision quantization that optimizes the balance between precision and efficiency for both weights and activations. This approach could involve more sophisticated methods for identifying salient parameters and adjusting the precision dynamically during inference. The focus could be on both accuracy and speedup.

**Efficient Serving and Deployment:**

- **Pipelined LLM Serving:** The lecture mentions the need for efficient LLM serving. `A thesis project could explore different pipeline designs for LLM serving. It could involve implementing and evaluating a new pipeline architecture, analyzing the trade-offs between latency and throughput, and optimizing for specific hardware characteristics (e.g., GPU architecture)`.

- **Dynamic Batching for Generative LLMs:** The discussion of continuous batching for generative LLMs suggests further research opportunities. `A thesis could develop and evaluate different approaches for dynamic batching strategies, considering varying input lengths and response times. Experimenting with different scheduling policies could be crucial.`


**Sparsity and Pruning Techniques:**

- **Context-Aware Pruning:** The concept of DejVu, context-aware sparsity, is presented. `A thesis could develop a new or improve an existing method for context-aware pruning of LLM parameters, focusing on maintaining model accuracy while reducing parameter count.`

- **Combined Pruning and Quantization:** Combining these techniques could further reduce the parameter footprint of LLMs. `A thesis project could investigate a novel strategy that combines pruning and quantization techniques, aiming to maximize accuracy and efficiency.`


## `Lecture - 14`: **TO-DO**

## `Lecture - 15`: `Long-Context LLM`



**Long Context Handling:**

- **Improving DuoAttention:** The lecture details the DuoAttention technique. A research project could delve deeper into specific aspects. Potential areas include:

  - **Optimal Head Assignment:** Investigate more sophisticated methods for determining which attention heads should be retrieval heads and which should be streaming heads. This could involve learning these assignments during pretraining or developing more nuanced criteria for selecting heads.
  - **Dynamic Chunk Size:** DuoAttention uses fixed-size chunks. Explore how to dynamically adjust the chunk size based on the input or the attention patterns to optimize memory and computation.

- **Extending Long Context with Enhanced Query-Aware Sparsity:** The Quest technique suggests query-aware sparsity. `A research project could explore the further potential of query-aware strategies by incorporating different scoring mechanisms for selecting relevant information from the KV cache for the current token query or improving the speed of these selections.`

- **Hybrid Models for Long-Context:** The lecture briefly mentions hybrid models. `Developing a _new_ hybrid architecture could be a thesis project. This might involve combining the efficiency of streaming methods with the full context capabilities of other techniques (e.g., transformers) to develop an efficient and effective architecture for handling very long context lengths. Consider the specific layers and their operations to find better synergies.`

- **State-Space Models (SSMs):** The concept of using state-space models to handle long contexts is presented. Exploring how to adapt and extend this approach to a wider range of tasks and models could be an avenue for research.

**Specific Methods and Areas of Further Investigation:**

- **Improved RoPE Implementation:** The presentation emphasizes that RoPE (Rotary Position Embedding) can support extending the context length. `A research project might focus on developing more efficient RoPE implementations or comparing how variations in the RoPE method affect the performance of LLMs under long context conditions, potentially by optimizing the way the angular information is encoded.`

- **Long-Context Evaluation Benchmarks:** The "Needle in a Haystack" and "LongBench" benchmarks are mentioned. Exploring ways to design or adapt existing benchmarks for assessing long-context capabilities in LLMs might be fruitful. This could include creating new benchmarks that better simulate real-world use cases or exploring ways to more effectively evaluate the performance of the models across various benchmarks.

**Additional Considerations:**

- **Addressing the "Attention Sink" Phenomenon:** The slides discuss the "Attention Sink" problem. A thesis could focus on developing methods to mitigate or even eliminate the "Attention Sink" phenomenon.

- **Pre-training with Dedicated Attention Sink Tokens:** The slides suggest an idea for pre-training models with dedicated "attention sink" tokens. `A thesis could explore how to apply this technique, potentially investigating how to best determine the optimal placement of these tokens during the pre-training phase, and how to optimize the pre-training strategy to best use the token.`


## `Lecture - 16`: `Vision Transformer`

The slides on Vision Transformers (ViT) and Hybrid Autoregressive Transformers (HART) offer several thesis research possibilities.  Here are a few, categorized for clarity:

**Vision Transformer (ViT) Enhancements:**

* **Efficient Multi-Scale Linear Attention for High-Resolution Dense Prediction:**  The slides highlight the difficulty of applying ViT to high-resolution images due to the quadratic computational cost. `A thesis project could investigate how to adapt or develop a new multi-scale linear attention mechanism to make ViT more efficient for high-resolution images`.  This could involve analyzing and potentially optimizing existing multi-scale approaches for this specific application.

* **Sparsity-Aware ViT Training:** The "SparseViT" technique suggests exploring sparsity in ViT.  `A research project could investigate a more sophisticated approach to sparsity, potentially incorporating dynamic sparsity, where the degree of sparsity adapts based on the input or during the training process, to further optimize computation and/or memory usage while preserving accuracy`.


**Hybrid Autoregressive Transformer (HART) Improvements:**

* **Improving HART's Image Quality and Efficiency:**  HART is presented as a faster alternative to diffusion models.  A thesis project could investigate methods to further enhance the image quality of HART's generated images while retaining or potentially improving its speedup over diffusion models.  Possible areas include:
    * **Hybrid Tokenization Strategies:**  Exploring alternative approaches for discretizing visual tokens in the HART framework could produce higher quality images. A research direction might be to investigate new methods for encoding/quantizing visual information, or comparing how variations in the quantization method affect the performance of the model.
    * **Optimizing the Residual Diffusion Process:** The residual diffusion stage in HART is crucial for reconstruction. A research topic could focus on optimizing this process to better preserve fine details while enhancing speed.

* **HART Architecture for Specific Downstream Tasks:** The slides mention using HART for specific tasks, such as medical image segmentation and autonomous driving. A thesis project could focus on adapting the HART architecture to *one or more of these specific tasks*, investigating potential advantages or drawbacks, and evaluating performance in these domains.


**Additional Research Areas (Inspired by the Lecture):**

* **Contrastive Learning for ViT:** The lecture discusses using contrastive learning in the context of ViT models.  A thesis could delve deeper into contrastive learning strategies, potentially developing a new method for contrastive pre-training specifically optimized for ViT architectures.  Consider using a specific data augmentation scheme or investigating how different choices for positive and negative samples impact the results.

* **Self-Supervised Pretraining for ViT:** The lecture emphasizes the benefits of self-supervised pretraining.  `A thesis project could develop and evaluate a *novel* self-supervised learning approach for pretraining ViT models, potentially exploring new pretext tasks or optimizing an existing method. Consider different datasets and evaluate their effect on model performance.`


## `Lecture - 17`: `Efficient GAN, Video, and Point Cloud`

The lecture slides on efficient GANs, video understanding, and point cloud understanding present a rich set of potential research topics for a master's thesis.  Let's break down some promising ideas:


**Generative Adversarial Networks (GANs):**

* **Improving GAN Training with Limited Data:** The slides explicitly address the problem of GANs degrading with limited data. `A research project could focus on developing and evaluating new techniques for data augmentation, discriminator improvements, or loss functions to improve the stability and performance of GANs when training with small datasets.`

* **AnyCost GANs:** The "AnyCost GAN" architecture is designed for consistent performance at different resolutions/channels. `A research topic could analyze and improve the AnyCost GAN approach further, focusing on different applications and more general conditions.   Research could investigate different methods of multi-resolution aggregation, optimizing memory usage and balancing training stability with accuracy.`


**Video Understanding:**

* **Efficient `Video Understanding` with Temporal Shift Modules (TSM):** The Temporal Shift Module (TSM) is presented as an effective method for video understanding. `A master's thesis could explore and enhance the performance of TSM in more specific video recognition tasks, such as action detection, or compare TSM to different architectures, evaluating tradeoffs among accuracy, speed, and computational costs. This includes investigating ways to further enhance TSM's capability in diverse video recognition or understanding tasks, focusing on specific applications like scene understanding, video prediction for autonomous driving, or action recognition in very-low-power settings.`


**Point Cloud Understanding:**

* **Advanced Sparse Point-Voxel Convolution (SPVConv):** The slides mention SPVConv for efficient processing of point clouds.  `A thesis project could investigate how to further optimize SPVConv by developing innovative methods for voxelization, or different convolution strategies, to enhance speed and accuracy, perhaps focusing on specific point cloud applications like 3D object detection in autonomous driving, or 3D semantic segmentation for augmented reality.`

* **Multi-Sensor Fusion (BEVFusion):** The lecture highlights BEVFusion.  `A research project could focus on different methods for multi-sensor fusion (e.g., camera + LiDAR) in the bird's-eye view (BEV) representation.  This includes investigating the effectiveness of using different architectural designs and/or loss functions in this fusion task.`


**Specific and Practical Improvements:**

* **Transfer Learning with GANs:** `Investigate using transfer learning techniques with GANs, for example, to adapt a pre-trained image generation model for a specific application with a smaller dataset.  Explore ways to adapt existing architectural designs.`


* **Hybrid Approaches for Video and Point Cloud:**  `Investigate combining techniques from video understanding (e.g., TSM) with techniques for point clouds (e.g., SPVConv). This could result in a more holistic, end-to-end system for handling videos and point clouds.`



**Action Recognition:**

* **Efficient Action Recognition in Crowded Scenes:** Existing action recognition models often struggle with cluttered backgrounds and multiple, overlapping actions. `A research project could focus on developing a model that can identify and classify actions with high accuracy, even in challenging scenes with numerous people or objects.  This could involve exploring new attention mechanisms, better feature representations, or different model architectures`.

* **Multi-Modal Action Recognition:**  `Combining visual information with audio, text, or other sensory input can improve action recognition accuracy and robustness. Research on incorporating multimodal information, perhaps using transformer architectures that integrate different modalities, is a good avenue to explore.  Consider the best method of fusion`.

* **Real-time Action Recognition in Dynamic Environments:** Designing action recognition models that can accurately and quickly identify actions in dynamic scenes is crucial. `A research project could develop an architecture that processes video streams in real-time, potentially using methods like online learning or adaptive processing based on the rate of change within the video`.


**Video Generation:**

* **Generating Realistic Videos of Complex Actions:** Can a model generate videos of actions that are both realistic (physically plausible) and complex (e.g., a person performing several actions in a specific order, involving multiple objects or interactions)? `Research projects can focus on incorporating or adapting physics-based models into the video generation pipeline to increase the model's ability to generate more natural and complex actions`.

* **Improving the Efficiency of Video Generation:** Many video generation models are computationally expensive and slow.  `Research on improving generation speed by optimizing the architecture, using generative-model compression, using specialized hardware or introducing approximations might lead to faster generation speeds, making the process more interactive.  Consider methods to improve the efficiency of both training and inference.`


* **Action-Conditional Video Generation:**  `Develop a model that generates videos where specific actions are conditional on given input.  For example, inputting text describing an action ("a person jumping over a hurdle") to generate realistic videos of that action.  This research could incorporate methods for generating more natural motions or adapting existing techniques to support complex actions.`

* **Multi-Modal Video Generation:** `Generating videos that combine multiple modalities (e.g., visual and audio) or that are conditional on other inputs (e.g., text prompts, keystrokes)`.



