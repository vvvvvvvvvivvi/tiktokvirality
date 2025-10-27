# Modelling Pop Music Success in the TikTok Era

This repository contains the code and analysis for my machine learning project **“Modelling Pop Music Success in the TikTok Era”**, which explores how short-form video virality reshapes the popularity of pop songs.  
The project compares **pre-TikTok (2017–2018)** and **post-TikTok (2021–2022)** periods to evaluate whether **TikTok engagement** or **playlist placement** more strongly predicts Billboard chart success.

---

## Overview

This research aims to investigate the relative importance of different established factors in facilitating a song’s commercial success, operationalised as its appearance on the Billboard Hot 100 chart. Using Random Forest models trained on audio features, streaming metrics, and platform exposure data, the study evaluates the predictive power of musicality compared to marketing promotions in driving success in the digital music ecosystem. 
By comparing time periods before and after TikTok’s dominance, the analysis quantifies how the drivers of pop success have evolved in the digital era.

---

## Machine Learning Methods
The analysis applied multiple supervised learning models to predict a song’s Billboard chart success (binary: charted / not charted) based on audio, virality, and playlisting features.

Models used:
- Logistic Regression — baseline classifier to establish feature interpretability.
- Random Forest Classifier — ensemble model capturing nonlinear relationships between features (e.g., danceability × TikTok engagement).
- Gradient Boosting (XGBoost / LightGBM) — fine-tuned model for higher accuracy and feature importance ranking.

---

## Technologies Applied
- **Python** — data processing, modelling, and analysis  
- **pandas, NumPy** — data wrangling and feature engineering  
- **scikit-learn** — machine learning (regression, feature selection, model evaluation)  
- **matplotlib, seaborn** — exploratory data analysis and visualisation  
- **Spotipy (Spotify Web API)** — retrieval of audio features and playlist data  
- **TikTok / Tokboard / Chartmetric data** — social virality metrics  
- **billboard.py** — weekly Billboard Hot 100 chart positions  
- **Jupyter Notebook** — documentation and experimentation environment  
