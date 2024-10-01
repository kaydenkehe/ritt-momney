# Goals

- [x] Find a CXR generator with desirable properties (high fidelity encoder and decoder, rich latent space)
- [ ] ???

# Setup

- Place PLCO_Fine_Tuned_120419.pth from https://github.com/circ-ml/CXR-Age into assets/models (CXR-Age) into assets/models
- Place cheff_autoencoder.pt and cheff_diff_uncond.pt from https://github.com/saiboxx/chexray-diffusion (LDM) into assets/models
- Create conda environment - 'conda env create -n <name> -f environment.yml'

# Resources, Papers

- GVR: Proof of concept for latent space counterfactuals, main method inspiration
    - https://arxiv.org/abs/1804.04539

- CXR-Age: Predicts time-to-death from CXRs, primary validation model (we may also use CXR-LC, CXR-CVD)
    - https://pubmed.ncbi.nlm.nih.gov/33744131/
    - https://github.com/circ-ml/CXR-Age

- Cheff: LDM for CXR generation
    - https://arxiv.org/abs/2303.11224
    - https://github.com/saiboxx/chexray-diffusion
