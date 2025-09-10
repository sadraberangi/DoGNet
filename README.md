# Fuzzy-DoGNet: Lightweight and Interpretable Deep Neuro-Fuzzy System for Traffic Sign Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Paper DOI](https://img.shields.io/badge/DOI-10.1109/ACCESS.2025.3606963-blue.svg)](https://doi.org/10.1109/ACCESS.2025.3606963)

---

## 📖 Overview

**Fuzzy-DoGNet** is a lightweight and interpretable deep neuro-fuzzy system designed for **Traffic Sign Recognition (TSR)**.  
It integrates **learnable Difference-of-Gaussian (DoG) filters**, a **multi-scale feature pyramid**, and a **feature-map attention module** with an **Unstructured Neuro-Fuzzy Inference System (UNFIS)** classifier.  

This unique combination achieves **state-of-the-art accuracy (99.72% on GTSRB)** while remaining computationally efficient (~1.9M parameters), making it well-suited for **real-time, resource-constrained deployment** on embedded platforms such as NVIDIA Jetson Orin Nano.

---

## ✨ Key Features

- 🔹 **Learnable Difference-of-Gaussian (DoG) Filters** – Adaptive, edge-enhanced feature extraction.  
- 🔹 **Multi-Scale Feature Pyramid** – Ensures scale invariance and robustness to varying input conditions.  
- 🔹 **Feature-Map Attention (FMA)** – Improves feature aggregation across scales.  
- 🔹 **Unstructured Neuro-Fuzzy Inference System (UNFIS)** – Provides interpretability, uncertainty-awareness, and rule-based reasoning.  
- 🔹 **Lightweight Design** – Only ~1.98M parameters, outperforming much larger CNNs.  
- 🔹 **Real-Time Ready** – ~964 FPS inference speed on Jetson Orin Nano.  

---

## 📊 Performance Highlights

| Dataset | Accuracy (%) | Parameters |
|---------|--------------|------------|
| **GTSRB** | **99.72** | 1.98M |
| **BTSD**  | 99.05 | 1.98M |
| **TSRD**  | 97.79 | 1.98M |

- Robust under **blur, brightness, contrast, and rotations**, with accuracy consistently >98.7%.  
- Outperforms multiple state-of-the-art baselines in both efficiency and accuracy.


---

## 📈 Results

- **Accuracy on GTSRB**: 99.72%  
- **Latency on Jetson Orin Nano**: ~1.06 ms/image (~964 FPS)  
- **Cross-dataset performance**: 99.05% (BTSD), 97.79% (TSRD)  
- **Robustness**: Accuracy >98.7% under blur, brightness, contrast, and rotation perturbations  

---

## 📑 Citation

If you use this repository or find our work helpful, please cite our paper:

```bibtex
@ARTICLE{11153480,
  author={Berangi, Sadra and Parchamijalal, Mohammad Mahdi and Salimi-Badr, Armin},
  journal={IEEE Access}, 
  title={Fuzzy-DogNet: A Lightweight and Interpretable Deep Unstructured Neuro-Fuzzy System based on Band-Pass Filters for Traffic Sign Recognition}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Dogs;Accuracy;Real-time systems;Attention mechanisms;Uncertainty;Neural networks;Computer architecture;Robustness;Lighting;Deep Fuzzy Neural Networks;Traffic Sign Recognition (TSR);Difference of Gaussian (DoG) Filters;Interpretability;Lightweight Neural Networks},
  doi={10.1109/ACCESS.2025.3606963}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
