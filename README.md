# CROCODILE 🐊: Causality aids RObustness via COntrastive DIsentangled LEarning

This project is the code base for our [**MICCAI 2024 paper**](https://arxiv.org/abs/2408.04949) at the 6th international workshop **Uncertainty for Safe Utilization of Machine Learning in Medical Imaging** ([UNSURE2024](https://unsuremiccai.github.io/)). We propose a new deep learning framework to tackle domain shift bias on medical image classifiers and improve their out-of-distribution (OOD) performance, fostering domain generalization (DG).

:computer: Go to [Get started](https://github.com/gianlucarloni/crocodile/tree/main#get-started-with-the-coding), if you want to use our code straight away. 

:star: Go to [Cite](https://github.com/gianlucarloni/crocodile/tree/main?tab=readme-ov-file#cite) to get our citation format to include in your work, if you've found it useful.

## Abstract

*Due to domain shift, deep learning image classifiers perform poorly when applied to a domain different from the training one. For instance, a classifier trained on chest X-ray (CXR) images from one hospital may not generalize to images from another hospital due to variations in scanner settings or patient characteristics. In this paper, we introduce our CROCODILE framework, showing how tools from causality can foster a model’s robustness to domain shift via feature disentanglement, contrastive learning losses, and the injection of prior knowledge. This way, the model relies less on spurious correlations, learns the mechanism bringing from images to prediction better, and outperforms baselines on out-of-distribution (OOD) data. We apply our method to multi-label lung disease classification from CXRs, utilizing over 750000 images from four datasets. Our bias-mitigation method improves domain generalization, broadening the applicability and reliability of deep learning models for a safer medical image analysis.*

## Impact

*The results of our in-distribution (ID) and out-of-distribution (OOD) data investigations reveal our method is behind its ablated versions and the competing method on i.i.d. data (ID) while is the best-performing model on the external never-before-seen data (OOD). This important result points to a necessary trade-off between in-domain accuracy and out-of-domain robustness on real-world data, supporting recent work. Notably, our method is the most effective in reducing the ID-to-OOD drop in performance. By leveraging causal tools, disentanglement, contrastive learning, and prior knowledge, it learns a better mechanism from image to prediction, relies less on spurious correlations, and breaks the boundaries across domains. Our bias-mitigation proposal is general and can be applied to tackle domain shift bias in other computer-aided diagnosis applications, fostering a safer and more generalizable medical AI.*

## Get started with the coding

We developed our training and testing scripts to fully leverage the multi-node multi-gpu infrastructure of the [LEONARDO Supercomputer](https://en.wikipedia.org/wiki/Leonardo_(supercomputer)). Since the execution of jobs on that system is handled by the SLURM workload manager, we'll make use of two utility bash scripts to submit our training and testing experiments.

Namely, the script [sbatch_submit.sh](https://github.com/gianlucarloni/crocodile/blob/main/sbatch_submit.sh) is where you specify the 'sbatch' command with custom parameters according to your system (e.g., --nodes, --ntasks-per-node, --cpus-per-task, --gpus-per-node, etc.).
As you can see, the 'sbatch' command itself launches the other bash script called [run_train.sh](https://github.com/gianlucarloni/crocodile/blob/main/run_train.sh), which is where the python script for training is executed with the desired input arguments.

### Note

Even if your setup does not include the usage of SLURM-based queues or multi-node multi-gpu experiments, you are still able to utilize our method! Indeed, you just need to install our [:wrench: requirements.txt ](https://github.com/gianlucarloni/crocodile/blob/main/code/requirements.txt) and slightly adapt the main python script (e.g., select a specified GPU device ID, not using SLURM variables, etc.). A tutorial guide on this is in process. Please, bear with us or just try adapting yourself. If encountering any issues or if you just want some curiosities clarified, open an Issue on this project or e-mail me (address can be found on the paper).

## Cite

If you found this code useful for your research, please cite our paper and star this repo. 

(work in progress): You can already access our preprint on ArXiv, although it is the Submitted Manuscript (pre peer-review). I will update citation upon publication of the final, camera-ready version of our paper, where we were given one extra page and improved the details and discussion about our methods and findings.
```
@article{carloni2024crocodile,
  title={CROCODILE: Causality aids RObustness via COntrastive DIsentangled LEarning},
  author={Carloni, Gianluca and Tsaftaris, Sotirios A and Colantonio, Sara},
  journal={arXiv preprint arXiv:2408.04949},
  year={2024}
}
```

### 
*Acknowledgement*. Some previous implementations and codes inspired us to develop this project and served as a basis for the present code to a varying extent:
- Transformer and backdoor general idea from this zc2024's [preliminary code](https://github.com/zc2024/Causal_CXR)
- Causal Medimg from my previous [repo](https://github.com/gianlucarloni/causal_medimg),
- _Causality map_ and _Causality factor extractor_ from our previous work [(Carloni et al., 2024)](https://doi.org/10.1016/j.eswa.2024.123433) and repo [gianlucarloni/causality_conv_nets](https://github.com/gianlucarloni/causality_conv_nets).
- Multi-node multi-gpu, data parallelism, and distributed training from this [gist.github.com](https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904) and this [www.idris.fr](http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html). 

We also acknowledge the CINECA award under the ISCRA initiative, for the availability of high-performance computing resources and support.
