# Overview

UPDATE: See our research paper 'losocs_research_paper_cs2822r.pdf'.

This project explores a new method for generating counterfactual images for regression models
by optimizing over representations in the latent space. The approach links the output of an
autoencoder, which models the input space of a target model, to the target model itself. By
optimizing the autoencoder’s latent space representation for a specific input, we guide the
generated images toward a desired output value, ideally producing a continuous sequence of
images which shows a human-interpretable change in the original input. We call our method
LoSoCs, or Latent Optimization Stream of Counterfactuals. We also propose and implement
metrics to quantitatively evaluate the quality of the generated streams of counterfactuals and
explore other aspects of the behavior of the process in our latent space. See the full research
paper developed for CS2822R here: https://drive.google.com/drive/folders/1ijwLrzKOkEZLFJ-e68XqekCslFxpfOcX?usp=sharing
# Setup

- Clone this repo
- Place PLCO_Fine_Tuned_120419.pth from https://github.com/circ-ml/CXR-Age (CXR-Age) into assets/models
- Place cheff_autoencoder.pt and cheff_diff_uncond.pt from https://github.com/saiboxx/chexray-diffusion (LDM) into assets/models
- Create a conda environment with `conda env create -n <name> -f environment.yml`
- Clone the repo at https://github.com/saiboxx/chexray-diffusion
    - Go into `chexray-diffusion/cheff/ldm/inference.py` and comment out line 46 (removing no_grad, so we can maintain the gradient properly during backprop)
    - Run `pip install -e ./` in the repo's top-level directory to install the package used to interface with the LDM

# Repo

- `assets/` contains all static assets. Helper scripts, all saved images, models, etc.
- `exploration/` contains exploratory, proof-of-concept, and tutorial notebooks. e.g., how to set up and use the LDM.
- `work/` contains more formal work. Actual attempts to tackle our research question.

# Resources, Papers

- GVR: Proof of concept for latent space counterfactuals, main method inspiration
    - https://arxiv.org/abs/1804.04539

- CXR-Age: Predicts time-to-death from CXRs, primary validation model (we may also use CXR-LC, CXR-CVD)
    - https://pubmed.ncbi.nlm.nih.gov/33744131/
    - https://github.com/circ-ml/CXR-Age

- Cheff: LDM for CXR generation
    - https://arxiv.org/abs/2303.11224
    - https://github.com/saiboxx/chexray-diffusion

