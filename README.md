# Machine Learning for Visual Acuity Estimation

🚧 **Work in progress**

This repository demonstrates an end-to-end machine learning workflow for visual acuity estimation. It covers:  
- data preprocessing  
- feature selection  
- model selection  
- stability analysis  
- model training and evaluation  
- deployment via FastAPI  

The original dataset contains sensitive medical information and is not publicly available.  

To improve reliability on out-of-distribution inputs, the API applies output constraints and returns warnings when post-processing rules are triggered.


## Project structure

```
visual-acuity-ml/  
├─ app/            # FastAPI application  
├─ artifacts/      # trained models and experiment artifacts  
├─ data/           # project datasets  
├─ notebooks/      # exploratory analysis and experiments  
├─ src/            # source code for preprocessing and ML workflows  
├─ README.md  
└─ requirements.txt  
```


## Installation

1. Clone the repository:

```bash
git clone https://github.com/molgan/visual-acuity-ml.git  
cd visual-acuity-ml
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```

