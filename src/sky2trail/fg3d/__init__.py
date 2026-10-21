"""ForestGen3D: generate TLS-like tree point clouds from airborne lidar.

A reproduction of Castorena et al., "From Canopy to Ground via ForestGen3D" (arXiv 2509.16346):
a conditional denoising diffusion model (DDPM) that turns the ALS points of one tree box into a
2048-point TLS-like cloud. The denoiser follows CGNet from Point Diffusion-Refinement
(Lyu et al. 2021): a PointNet++ U-Net on the noisy points, a PointNet++ encoder on the ALS
points, and a feature mapper that modulates the first with the second.

Everything is pure PyTorch (no custom CUDA kernels), so it runs on CUDA, Apple MPS and CPU.

numpy parts (no PyTorch needed): synthetic, metrics (numpy functions), pairs (building).
PyTorch parts: ops, diffusion, model, train, sample.
"""
