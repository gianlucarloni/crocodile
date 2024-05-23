# *CROCODILE* 🐊: *C*ausality aids *RO*bustness via *CO*ntrastive *DI*sentangled *LE*arning for generalizable and explainable AI

This project is the code base for our [paper](placeholder). Please, ...

## Summary (Abstract)

As its acronym suggests, with our method we make the following contributions:

- *Causality*: since we build on the causal theory and implement the backdoor adjustment to break the connection between the prediction target (e.g., disease label) and spurious features (e.g., lead letters in the corner of a chest X-ray image). Moreover, we propose to use the _causality map_ and _causality factor extractor_ from our previous work [(Carloni et al., 2024)](https://doi.org/10.1016/j.eswa.2024.123433) and repo [gianlucarloni/causality_conv_nets](https://github.com/gianlucarloni/causality_conv_nets) during the stratification and addition phase of the backdoor adjustment.
- *RObustness*: since our method make the model learn the true _causal_ features that determine the outcome, disregarding the spurious non-causal features
- *COntrastive*: since we frame our learning setting as a contrastive learning schema leveraging images from different source datasets. This way, we are able to impose crucial consistency/similarity constraints to the latent representation of instances (i) coming from the same dataset and with same diagnosis label, (ii) coming from the same dataset but with different diagnoses, (iii) sharing the same diagnosis label but coming from different datasets, and (iv) coming from different datasets and with different diagnoses.
- *DIsentangled*: since we build on the multi-head attention meccanism of the Transformer and modify it to return a set of attention weights _A_ along with its negated _(1-A)_. We'd use the former to compute the causal features and the latter to compute the non-causal (spurious) features. This way, we attain feature disentanglement with a more semantically defined latent space.
- *generalizable*: since we showed how our method helps in improving Domain Generalization (DG): when the data distribution changes from training and deployment scenarios, the model can behave more consistently across new data.
- *explainable*: since ...
 

## Get started with the coding

We developed our training and testing scripts to fully leverage the multi-node multi-gpu infrastructure of the [LEONARDO Supercomputer](https://en.wikipedia.org/wiki/Leonardo_(supercomputer)). Since the execution of jobs on that system is handled by the SLURM workload manager, we'll make use of two utility bash scripts to submit our training and testing experiments.

Namely, the script [sbatch_submit.sh](https://github.com/gianlucarloni/crocodile/blob/main/sbatch_submit.sh) is where you specify the 'sbatch' command with custom parameters according to your system (e.g., --nodes, --ntasks-per-node, --cpus-per-task, --gpus-per-node, etc.).
As you can see, the 'sbatch' command itself launches the other bash script called [run_train.sh](https://github.com/gianlucarloni/crocodile/blob/main/run_train.sh), which is where the python script for training is executed with the desired input arguments.

### Note

Even if your setup does not include the usage of SLURM-based queues or multi-node multi-gpu experiments, you are still able to utilize our method! Indeed, you just need to ...

### Baseline

## Acknowledgement 

Some previous implementations and codes inspired us to develop this project and served as a basis for the present code to a varying extent:
- Transformer and backdoor general idea from this zc2024's [preliminary code](https://github.com/zc2024/Causal_CXR)
- Causal Medimg from my previous [repo](https://github.com/gianlucarloni/causal_medimg),
- _Causality map_ and _Causality factor extractor_ from our previous work [(Carloni et al., 2024)](https://doi.org/10.1016/j.eswa.2024.123433) and repo [gianlucarloni/causality_conv_nets](https://github.com/gianlucarloni/causality_conv_nets).
- Multi-node multi-gpu, data parallelism, and distributed training from this [gist.github.com](https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904), and this [www.idris.fr](http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html). 

We also acknowledge the CINECA award under the ISCRA initiative, for the availability of high-performance computing resources and support

## Cite

Please cite our [paper](placeholder) if you found this code useful for your research:

```
@article{carloni2024,
}
```
