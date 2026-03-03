# ATDMDiff，Affinity-Target Dual-Driven Diffusion Model

## Install conda environment via conda yaml file
```bash
conda env create -f environment.yaml
```
## Training
```bash
python train.py --config configs/ATAMDiff.yml
```
## Sampling
Modify each sample in the test set 100 times.
```bash
bash sample.sh
```
## Evaluation
Run the evaluation script after the sampling process is complete.
```bash
bash evaluate.sh
```
