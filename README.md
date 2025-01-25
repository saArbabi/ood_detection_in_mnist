# OOD Detection with MNIST

This repository is focused on building a model for out-of-distribution (OOD) detection using the MNIST dataset. The project is a work-in-progress.

---
### Running the Code

To train the model and save the initial checkpoints, use the following command:

```bash
python src/train.py --dry-run --save-model --epochs=5
```
---
### TODO List

- [ ] Visualize model embeddings.
- [ ] Test the model on OOD data and visualize changes in the embedding space.
- [ ] Improve model robustness to OOD by enhancing representation quality and separating IID from OOD data.
- [ ] Visualize how embeddings transform during training iterations.
 