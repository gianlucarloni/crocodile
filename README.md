# CROCODILE 🐊 CAusality aids RObustness via COntrastive DIsentangled LEarning for generalizable and explainable AI

This project is the code base for our [paper](placeholder). Please, ...

## Summary (Abstract)

As the name suggests, our method makes the following contributions:

- *CAusality*: since we build on the causal theory and implement the backdoor adjustment to break the connection between the prediction target (e.g., disease label) and spurious features (e.g., lead letters in the corner of a chest X-ray image)
- *RObustness*: since our method make the model learn the true _causal_ features that determine the outcome, disregarding the spurious non-causal features
- *COntrastive*: since we frame our learning setting as a contrastive learning schema leveraging images from different source datasets. This way, we are able to impose crucial consistency/similarity constraints to the latent representation of instances (i) coming from the same dataset and with same diagnosis label, (ii) coming from the same dataset but with different diagnoses, (iii) sharing the same diagnosis label but coming from different datasets, and (iv) coming from different datasets and with different diagnoses.
- *DIsentangled*: since we build on the multi-head attention meccanism of the Transformer and modify it to return a set of attention weights _A_ along with its negated _(1-A)_. We'd use the former to compute the causal features and the latter to compute the non-causal (spurious) features. This way, we attain feature disentanglement with a more semantically defined latent space.
- *generalizable*: since we showed how our method helps in improving Domain Generalization (DG): when the data distribution changes from training and deployment scenarios, the model can behave more consistently across new data.
- *explainable*: since ...
 

## Get started with the coding

We developed our training and testing scripts to fully leverage the multi-node multi-gpu infrastructure of the [LEONARDO Supercomputer](https://en.wikipedia.org/wiki/Leonardo_(supercomputer)) 
