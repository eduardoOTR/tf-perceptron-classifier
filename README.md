# TensorFlow Perceptron Classifier

A single-layer Perceptron implemented with TensorFlow and Keras for the automatic classification of two petroleum-oil purity classes.

## Overview

This project classifies oil samples obtained from a fractional distillation process. Classification is based on measurements of three physicochemical properties. The model learns a linear decision boundary that separates the two purity classes.

### Class convention

- **P1:** represented by the target value `-1`
- **P2:** represented by the target value `1`

## Model architecture

The classifier is a single-neuron Perceptron with three inputs (`x₁`, `x₂`, and `x₃`), a bias input of `-1`, and the associated bias weight `θ = w₀`. The weighted inputs are combined by a linear summation unit `u`, followed by a linear activation function that produces the output `y`.

![Single-layer Perceptron architecture](assets/perceptron.png)

The neuron computes a linear response of the form:

```text
u = w₀(-1) + w₁x₁ + w₂x₂ + w₃x₃
```

and uses the linear activation function to generate `y`.

## Summary

| Parameter | Specification |
| --- | --- |
| Model | Single-layer Perceptron |
| Inputs | `x₁`, `x₂`, `x₃` |
| Net input | `u` |
| Output | `y` | 
| Classes | `-1` (P1) and `1` (P2) |
| Learning rate | `0.05` |
| Optimizer | Adam |
| Framework | TensorFlow and Keras |

The repository also features custom convergence callbacks to monitor training and identify when the model has converged.
