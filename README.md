# Machine Learning for Visual Acuity Estimation

🚧 **Work in progress**

This project demonstrates an end-to-end machine learning workflow for estimating pre-operative visual acuity from clinical parameters. The repository covers the full lifecycle, including data preprocessing, model training, evaluation, and deployment via a FastAPI service.

The original dataset contains sensitive medical information and is not publicly available.  
A synthetic dataset with the same schema and similar statistical properties is provided for demonstration purposes.

To improve reliability on out-of-distribution inputs, the API applies output constraints and returns warnings when post-processing rules are triggered.


## Project structure

```
visual-acuity-ml/  
├─ app/            # FastAPI application  
├─ artifacts/      # trained models  
├─ data/           # synthetic dataset for demonstration  
├─ notebooks/      # exploratory analysis and experiments  
├─ src/            # source code for data processing and model training  
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

