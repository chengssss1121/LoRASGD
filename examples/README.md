# Executed example data

All data used in the bundled example runs are synthetic. None of these curves
are a completed CIFAR-10 experiment or a reproduction of the reference figures.

- `logistic_synthetic_sweep/`: 3 learning rates x 3 seeds, 960 updates each.
- `figures_logistic_synthetic/`: exact spectral A/B/sum curves, seed means and
  one-sample-standard-deviation bands; raw individual trajectories remain saved.
- `resnet_fake_smoke/`: two SGD updates on torchvision FakeData; 21 LoRA pairs.
- `figures_resnet_fake/`: corresponding multi-pair aggregates and layer heatmap.
- `fake_resnet18_features.npz`: 16 training and 8 test synthetic images passed
  through a frozen randomly initialized ResNet-18, yielding 512D features.
- `cached_feature_pipeline_smoke/`: classifier trained using that feature file.
- `theory_step_smoke/`: 16 updates using the conservative fixed-horizon step.
- `figures_theory_step/`: corresponding diagnostic figure with a fixed C_T line.

The ResNet smoke run's full model checkpoint is omitted from the compact source
archive. Running `train_resnet.py` creates its own full checkpoint. The recorded
CSV measurements, saved configuration, evaluation data, and summary are included.
