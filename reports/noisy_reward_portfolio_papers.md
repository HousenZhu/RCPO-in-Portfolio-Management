# Papers Related To Noisy Rewards, Noisy Data, And Robust Portfolio RL

This file collects papers related to noisy reward, noisy financial data, reward smoothing, adversarial perturbation, and robust portfolio optimization. The main research framing is:

```text
Most portfolio RL papers handle noisy market states or noisy return estimates.
Most reward-noise RL papers handle perturbed reward channels, but not portfolio management.
This project can connect these two:
test DRC/GDRC in portfolio RL when the training reward is corrupted,
while evaluation still uses clean realized portfolio return.
```

## Closest To The GDRC Idea

### The Distributional Reward Critic Framework for Reinforcement Learning Under Perturbed Rewards

Link: [https://arxiv.org/html/2401.05710v3](https://arxiv.org/html/2401.05710v3)

This is the main theoretical anchor for DRC/GDRC. It directly studies reinforcement learning when observed rewards are perturbed, corrupted, or noisy, and proposes DRC/GDRC to recover useful reward learning. It is not portfolio-specific, but it gives the clean reward-perturbation framework.

### Contrastive Learning and Reward Smoothing for Deep Portfolio Management

Link: [https://www.ijcai.org/proceedings/2023/441](https://www.ijcai.org/proceedings/2023/441)

This is one of the closest portfolio-specific papers. It does not add random reward noise exactly, but it argues that training directly on per-period returns is difficult because financial returns are unpredictable. It uses reward smoothing as regularization so the RL model does not chase immediate uncertain profits.

## Noisy Financial Data In Portfolio RL

### A Novel Anti-Risk Method for Portfolio Trading Using Deep Reinforcement Learning

Link: [https://www.mdpi.com/2079-9292/11/9/1506](https://www.mdpi.com/2079-9292/11/9/1506)

This paper is relevant because it says classical RL methods often ignore exogenous noise in financial time series. It uses a stacked sparse denoising autoencoder plus A2C. This supports the idea that noisy financial signals can hurt portfolio RL and that denoising or correction is a valid research angle.

### Reinforcement-Learning Based Portfolio Management with Augmented Asset Movement Prediction States

Link: [https://arxiv.org/abs/2002.05780](https://arxiv.org/abs/2002.05780)

This paper discusses noisy, imbalanced heterogeneous asset information and market uncertainty. It augments the RL state with asset movement prediction signals to improve robustness.

### Portfolio Management Using Online Reinforcement Learning With Adaptive Exploration And Multi-Task Self-Supervised Representation

Link: [https://www.sciencedirect.com/science/article/abs/pii/S1568494625001577](https://www.sciencedirect.com/science/article/abs/pii/S1568494625001577)

This paper discusses noisy signals, volatility, unrealistic simulation environments, and concept drift. It uses self-supervised representation learning and adaptive exploration to improve RL portfolio learning.

### Knowledge Distillation For Portfolio Management Using Multi-Agent Reinforcement Learning

Link: [https://www.sciencedirect.com/science/article/pii/S1474034623002240](https://www.sciencedirect.com/science/article/pii/S1474034623002240)

This paper is relevant because it frames financial markets as noisy and hard for stable RL training. It uses a student-teacher and multi-agent distillation setup to stabilize the learned trading strategy.

### A Novel Portfolio Selection Method Via Deep Reinforcement Learning

Link: [https://www.mdpi.com/2079-8954/14/3/292](https://www.mdpi.com/2079-8954/14/3/292)

This recent paper focuses on extracting reliable representations from non-stationary and noisy financial data, using a denoising-style module in a DRL portfolio-selection framework.

## Adversarial Or Perturbed Portfolio RL

### Adversarial Deep Reinforcement Learning In Portfolio Management

Link: [https://ideas.repec.org/p/arx/papers/1808.09940.html](https://ideas.repec.org/p/arx/papers/1808.09940.html)

This paper uses adversarial training in portfolio RL and reports improved training efficiency, Sharpe ratio, and return. It supports robustness training, though it focuses more on adversarial perturbation than reward noise.

### Adversarial Attacks Against Reinforcement Learning-Based Portfolio Management Strategy

Link: [https://doaj.org/article/e8d1c6aab2c84bf8ad6110b9260dc27a](https://doaj.org/article/e8d1c6aab2c84bf8ad6110b9260dc27a)

This paper shows that RL-based portfolio strategies can be vulnerable to small adversarial perturbations. It supports the broader claim that portfolio RL should be tested under corrupted or noisy inputs.

### Safe-FinRL: A Low Bias And Variance Deep Reinforcement Learning Implementation For High-Frequency Stock Trading

Link: [https://ideas.repec.org/p/arx/papers/2206.05910.html](https://ideas.repec.org/p/arx/papers/2206.05910.html)

This paper is not reward-noise specific, but it is relevant to unstable RL estimation in finance. It addresses non-stationarity and bias/variance problems in high-frequency trading RL.

## Classical Portfolio Noise And Estimation Error

### Noisy Covariance Matrices And Portfolio Optimization II

Link: [https://www.sciencedirect.com/science/article/pii/S0378437102014991](https://www.sciencedirect.com/science/article/pii/S0378437102014991)

This is part of a classic line of work showing that empirical covariance matrices contain substantial noise, which damages portfolio optimization.

### Portfolio Optimization With Noisy Covariance Matrices

Link: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3595423](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3595423)

This paper directly studies how noisy covariance estimates lead to under-forecasted risk, higher out-of-sample volatility, leverage, turnover, and inefficient risk allocation.

### Portfolio Selection With Robust Estimation

Link: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=911596](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=911596)

This paper shows that portfolio weights can be unstable under estimation error and proposes robust estimation methods.

### Estimation Error And Portfolio Optimization: A Resampling Solution

Link: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2658657](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2658657)

This is useful background for the idea that optimized portfolios are highly sensitive to noisy estimates. It is not reinforcement learning, but it is foundational motivation.

### Distributionally Robust Portfolio Optimization

Link: [https://pmc.ncbi.nlm.nih.gov/articles/PMC7956065/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7956065/)

This paper treats the return distribution itself as uncertain and optimizes against a set of plausible distributions. This is a strong theoretical neighbor to training under noisy or uncertain reward and return signals.

## Market Microstructure Noise

### High Frequency Market Microstructure Noise Estimates And Liquidity Measures

Link: [https://www.nber.org/papers/w13825](https://www.nber.org/papers/w13825)

This paper is useful for explaining why observed prices and returns can be noisy in real markets, especially at high frequency.

### High-Dimensional Minimum Variance Portfolio Estimation Based On High-Frequency Data

Link: [https://www.sciencedirect.com/science/article/pii/S0304407619301630](https://www.sciencedirect.com/science/article/pii/S0304407619301630)

This paper studies portfolio estimation from high-frequency returns that may be contaminated by microstructure noise.

## Suggested Project Framing

A strong way to position the current project is:

```text
Clean synthetic portfolio rewards are an idealized setting.
In real trading, the reward channel can be corrupted by execution uncertainty,
transaction-cost estimation error, stale prices, slippage, bid-ask bounce,
and short-window risk-estimation noise.

This project tests whether DRC/GDRC can improve portfolio RL when the training
reward is noisy, while evaluation remains based on clean realized portfolio return.
```

