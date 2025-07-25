# 🚀 Hybrid Reinforcement Learning for Financial Fraud Detection

This repository contains the code and implementation for the research paper: **"Decoupling Supervised Learning and Policy Optimization: A Hybrid Framework for Financial Fraud Detection"**.

The project introduces a novel hybrid framework that uses Offline Reinforcement Learning (RL) to make smarter, cost-sensitive decisions for flagging financial transactions.

---

## Table of Contents
* [The Problem](#the-problem-)
* [Our Solution: A Hybrid Approach](#our-solution-a-hybrid-approach-)
* [Key Results](#key-results-)
* [System Architecture](#system-architecture-)

---

## The Problem 😕
Traditional fraud detection models are good at one thing: **classifying** if a transaction *looks like* past fraud. They work like simple spam filters. However, they fall short in two critical ways:

1.  **They don't understand costs.** A missed fraud (False Negative) is thousands of times more costly than a wrongly blocked transaction (False Positive). Standard models trained on metrics like accuracy can't grasp this trade-off.

2.  **They are reactive, not adaptive.** They answer "*Is this risky?*" but not the real business question: "*What is the best **action** to take, given the risk and costs?*"

---

## My Solution: A Hybrid Approach 💡
I reframe fraud detection from a simple classification task to a **sequential decision-making problem**. This framework decouples the system into two specialized parts:

1.  **The Perception Model (The "Eyes")**: A state-of-the-art LightGBM model analyzes over 400 features for each transaction to distill a single, powerful `risk_score`. It excels at pattern recognition.

2.  **The Decision Agent (The "Brain")**: An ensemble of Conservative Q-Learning (CQL) agents receives this `risk_score` and other key context. Instead of just classifying, it learns an optimal *policy* to decide whether to **Approve** or **Flag** a transaction to maximize a reward function that balances the high cost of fraud against the friction of false alarms.

This hybrid model learns not just to predict risk, but to make the most profitable and efficient decision at every step.

*High-level view of the five-stage pipeline.*

---

## Key Results 📊
I tested this framework on the massive and realistic IEEE-CIS Fraud Detection dataset. My hybrid RL approach decisively outperformed a powerful, highly-tuned supervised LightGBM baseline on the metrics that matter most.

### Head-to-Head Performance
While both models had a similar ability to rank transactions (ROC-AUC),the RL model was vastly superior at actually identifying fraud in a practical setting.

| Metric | V-Final: RL Ensemble 🧠 | V0: Supervised Baseline 👀 | Improvement |
| :--- | :---: | :---: | :---: |
| **PR-AUC** | **0.807** | 0.036 | **+2141%** |
| **Optimal F1-Score** | **0.77** | 0.71 | **+8.5%** |
| **Optimal Precision**| **0.83** | 0.75 | **+10.7%**|
| **ROC-AUC** | 0.958 | 0.961 | - |

The **22x improvement in PR-AUC** shows the RL model's vastly superior ability to handle the severe class imbalance and find fraudulent transactions effectively.

### Analysis of Disagreements: Proving Superior Decision-Making
I isolated the 8,447 test transactions where the RL model and the baseline disagreed. The results were clear:

* **Catching Missed Fraud**: For fraudulent transactions the baseline was about to **approve**, the RL agent correctly overturned the decision and **flagged them with 99% recall**. It acts as a critical safety net.

* **Reducing False Alarms**: For legitimate transactions the baseline was about to **block**, the RL agent correctly overturned the decision and **approved them 66% of the time**, reducing customer friction and operational costs.

> In essence, when the models disagreed, the RL agent's policy made the more intelligent, value-driven decision **78% of the time**.

---

## System Architecture ⚙️
The end-to-end pipeline consists of five main stages:

1.  **Feature Engineering**: Create new, context-rich features from the raw data, such as user spending habits (`amt_vs_card1_mean`) and transaction velocity (`time_since_last_tx`).

2.  **Data Preprocessing**: Clean the data by handling missing values, encoding categorical features, and applying standard scaling.

3.  **Risk Score Distillation**: Use a 5-fold cross-validation strategy to train a LightGBM model. This generates a leak-proof `risk_score` for every transaction in the dataset, which serves as the primary signal for the RL agent.

4.  **Offline RL Policy Training**: Frame the task as a Markov Decision Process. Train an ensemble of 5 independent Conservative Q-Learning (CQL) agents on the historical data using a custom reward function that penalizes missed fraud and false alarms.

5.  **Ensemble Prediction**: Average the fraud probabilities from the 5 trained agents and apply a final threshold (optimized for the F1-score) to make the final `Approve` / `Flag` decision.
