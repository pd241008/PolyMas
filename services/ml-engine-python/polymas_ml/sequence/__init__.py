"""System B: Mamba sequence-model pipeline on per-patient reference sequences.

Mirrors System A's synthetic patients (same seed 42 PRS genotypes, same
disease labels, same train/calibration split) but represents each patient as a
nucleotide sequence (Ensembl reference windows with genotype injection) instead
of a tabular PRS feature matrix.
"""
