# Hubacek2019-soccer-prediction-replication

### Overview

This project is a **replication of the best-performing XGBoost model** from Hubáček et al. (2018/2019), “Learning to predict soccer results from relational data with gradient boosted trees.”  
- [Official paper (Machine Learning, Springer)](https://doi.org/10.1007/s10994-018-5704-6)

The replication **focuses exclusively on the English Premier League**, using public datasets from [football-data.co.uk](https://www.football-data.co.uk/) for the seasons 2000–2025/26.

---

### Features

- **Full replication** of the paper’s all feature categories + XGBoost approach.
- Adapted to work on **Premier League data only**, though should also work with other single leagues with minimal changes.
- **Basic dataset preprocessing and loading** for 2000/01–2025/26 EPL matches.
- Results can be compared directly to those reported in the original paper.

---

### Differences from the Original Paper

- **Dataset:**  
  This replication uses only EPL data from football-data.co.uk, not the full set of leagues used in the paper.
- **Scope:**  
  Covers EPL seasons from 2000/01 to 2025/26.
- **Implementation:**  
  Adapted for public datasets and simplified file handling for ease of use.

---

### Getting Started

1. Clone this repository.
2. Download EPL datasets (CSV) from [football-data.co.uk](https://www.football-data.co.uk/).
3. Place datasets in the appropriate folder (see code for expected path).
4. Install dependencies: (list not added yet).
5. Run preprocessing and model training scripts as needed.

---

### Citation

If you use this code or data for research, please cite the original paper:

> Hubáček O, Šourek G, Železný F (2019). Learning to predict soccer results from relational data with gradient boosted trees. *Machine Learning*, 108(1):29–47.
> https://doi.org/10.1007/s10994-018-5704-6
