# astro_augment

Do natural-image augmentation policies help galaxy morphology classification?

Astronomical deep learning usually restricts data augmentation to "physics-preserving" transformations:
flips, rotations and crops, which change the image but not the morphology. Natural-image computer vision
instead trains with strong, information-destroying policies. This repository tests whether those policies
beat the physics-preserving recipe of Walmsley et al. (2022) on Galaxy Zoo DECaLS (GZD-5).

The comparison covers:

- **Baseline:** the H/V flip + rotation + random crop recipe of Walmsley et al. (2022).
- **Policies:** RandAugment, AutoAugment, TrivialAugment, AugMix, CutMix and MixUp.
- **Crop modality:** each policy is applied after a plain random crop (`_rc`) and after the full Walmsley
  recipe (`_wt`).
- **Strength:** weak, standard and strong presets of each policy.
- **Training-set size:** 1% to 100% of the labelled galaxies.
- **AugMix with its Jensen–Shannon consistency loss:** an additional variant.

Every configuration is trained with 5 seeds.

**Model and loss:** an EfficientNet-B0 trained on the Galaxy Zoo vote counts, with the Dirichlet–multinomial
loss of Walmsley et al. (2022) over the 10 GZD-5 questions.

The training framework (models, tasks, transforms, the trainer) is
[optastra](https://github.com/Aureusa/optastra). This repository contains only what is specific to the
study.

## Repository layout

```
train.py                  single entrypoint for every experiment (one run = experiment x seed x fraction)
experiments.py            registry: experiment name -> train/val transforms, batch transforms, task
evaluate.py               re-evaluate the best checkpoints of the original full-data runs on the test split
make_splits.py            the fixed 80/10/10 train/val/test split
data/                     DECaLSDataset (reads the PNGs straight from the zip archives), collate functions
task/                     GZD5 decision tree, Dirichlet-multinomial task, AugMix + Jensen-Shannon task
transforms/               Walmsley transforms, CutMix/MixUp on vote counts, policy presets, AugMix views
model/                    EfficientNet-B0 built with optastra
tests/                    pytest unit tests for the loss functions
*.slurm                   Slurm job scripts: training-set-size sweep, AugMix + JS, re-evaluation
info_destroyed.py         estimate of how much of each image an augmentation destroys (used by the analysis notebook)
prune_old_checkpoints.py  delete per-epoch checkpoints of evaluated runs, keeping the best one

augmentation_results.ipynb  analysis: statistics, claims, tables (writes results/tables/)
paper_figures.ipynb         the paper figures (reads results/tables/, writes results/figures/paper/)
augmentation_gallery.ipynb  one galaxy under every augmentation
dataset_overview.ipynb      split sizes and morphology tables
```

Outputs are written to `runs/` (checkpoints and logs), `results/` (tables and figures) and `slurm_output/`.
None of these are under version control.

## Getting started

### 1. Install optastra and the dependencies

astro_augment imports optastra, so install optastra first, as an editable package, into the same
environment:

```bash
python -m venv ~/.envs/astro_augment
source ~/.envs/astro_augment/bin/activate

git clone https://github.com/Aureusa/optastra.git
pip install -e optastra

git clone https://github.com/Aureusa/astro_augment.git
cd astro_augment
pip install -r requirements.txt
```

Python ≥ 3.10 and PyTorch ≥ 2.1 are required. Training needs a GPU. On a cluster, install inside a job or
interactive session, not on a login node.

### 2. Get the data

Download the Galaxy Zoo DECaLS release of Walmsley et al. (2022) from Zenodo: the GZD-5 volunteer table and
the four DR5 PNG zip archives. Put them in one directory:

```
$GZD5_DATA_DIR/
    gz_decals_volunteers_5.parquet      # the volunteer table (convert the released CSV with pandas if needed)
    gz_decals_dr5_png_part1.zip ... gz_decals_dr5_png_part4.zip
```

Then point the code at it and create the splits:

```bash
export GZD5_DATA_DIR=/path/to/gz_decals     # default: ~/project_data/aa_gz_decals
python make_splits.py                       # -> decals_split_{train,val,test}.parquet in $GZD5_DATA_DIR
```

The split keeps galaxies with at least 3 votes on the first question, giving 249,581 galaxies. It is
stratified by the majority answer to that question and seeded, so it is the same on every machine.

### 3. Check the installation

```bash
pytest                                      # unit tests, CPU only, no data needed
python train.py --list                      # every registered experiment
```

### 4. Train

One run:

```bash
python train.py --experiment walmsley_transforms --seed 42 --train-fraction 0.05 --val-size 5000
python train.py --experiment augmix_wt         --seed 42 --train-fraction 0.05 --val-size 5000
```

Each run writes `runs/sweep/frac_<f>/<experiment>_<seed>/`, containing:

- the config;
- the indices of its training subset;
- the best checkpoint (lowest validation loss);
- `DONE.json` with the test metrics.

Rerunning the same command resumes an unfinished run and skips a finished one. Training subsets are fixed
and nested across fractions, and they are the same for every seed. `python train.py --help` lists the
options.

Experiment names follow `<policy>[_weak|_strong]_<modality>`, with modality `rc` (random crop) or `wt`
(H/V flip + rotation + random crop). Examples: `trivial_augment_wt`, `cutmix_strong_rc`,
`augmix_jsd_wt`. The reference runs are `walmsley_transforms`, `baseline_rc` (random crop only) and
`baseline_B` (centre crop, no augmentation).

Whole studies run as Slurm jobs. Submit them from the repository root:

```bash
mkdir -p slurm_output
sbatch train_sweep.slurm          # every policy x 5 seeds x 1, 5, 20, 50% of the training set
sbatch train_augmix_jsd.slurm     # AugMix + Jensen-Shannon loss: sweep, plus the strengths on the full data
```

The scripts load ALICE (Leiden) modules and activate `~/.envs/astro_augment`. Set `VENV=/path/to/venv`, and
edit the `module load` and `#SBATCH` lines, for another cluster.

### 5. Analyse

Run `augmentation_results.ipynb` first, which aggregates every run into `results/tables/`. Then run
`paper_figures.ipynb`.

## Citation

If you use this code, please cite the accompanying paper (in preparation) and Walmsley et al. (2022),
MNRAS 509, 3966, for the data and the baseline recipe.
