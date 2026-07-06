# Peer-to-Peer Energy Trading in Dairy Farms using Multi-Agent Reinforcement Learning

## Overview

This repository contains the implementation of the research paper **"Peer-to-Peer Energy Trading in Dairy Farms using Multi-Agent Reinforcement Learning"** by Mian Ibad Ali Shah, Marcos Eduardo Cruz Victorio, Maeve Duffy, Enda Barrett, and Karl Mason. The study explores decentralized energy management in rural dairy farming communities using **Peer-to-Peer (P2P) energy trading** combined with **Multi-Agent Reinforcement Learning (MARL)** methods such as **Proximal Policy Optimization (PPO)** and **Deep Q-Networks (DQN)**.

The goal is to optimize energy distribution in dynamic environments, improve energy efficiency, reduce electricity costs, minimize peak hour demand, and increase electricity revenue. This repository provides Python/PyTorch-based implementations of the algorithms and mechanisms discussed in the paper.

---

## Core Concept

### Peer-to-Peer Energy Trading
P2P energy trading allows individual energy producers and consumers within a community to directly exchange energy. This approach is particularly suited for decentralized energy systems in rural areas, such as dairy farms, where renewable energy sources like solar panels and wind turbines are often used.

### Multi-Agent Reinforcement Learning (MARL)
Traditional rule-based energy management systems can struggle to adapt to dynamic and unpredictable environments. MARL leverages multiple agents working collaboratively to learn optimal policies for energy trading and management. The two key algorithms used in this study are:
- **Deep Q-Networks (DQN):** A value-based method that uses deep learning to approximate Q-values for state-action pairs.
- **Proximal Policy Optimization (PPO):** A policy gradient method that improves policy optimization stability and efficiency.

### Key Modules
1. **Auction-Based Market Clearing:** A mechanism for determining energy prices and matching buyers and sellers in P2P trading.
2. **Price Advisor Agent:** An intelligent agent that advises participants on optimal trading prices.
3. **Load and Battery Management:** Ensures efficient utilization of energy storage systems and regulates energy consumption.

---

## Repository Structure

```
.
├── datasets/                 # Contains sample energy demand and generation data for Ireland and Finland
├── src/                      # Source code for the implementation
│   ├── agents/               # Implementation of PPO and DQN reinforcement learning agents
│   ├── environment/          # Simulation environment for the P2P energy trading system
│   ├── auction_mechanism/    # Auction-based market clearing logic
│   ├── price_advisor/        # Price advisor agent implementation
│   └── utils/                # Utility functions for data processing and visualization
├── results/                  # Precomputed performance metrics and visualizations
├── configs/                  # Configuration files for experiments
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # License information
```

---

## Key Features

1. **Simulation Environment:**
   - Models P2P energy trading in dairy farms using renewable energy sources and storage systems.
   - Incorporates dynamic energy demand and generation data from real-world datasets in Ireland and Finland.

2. **Multi-Agent Reinforcement Learning:**
   - **DQN Implementation:** Trains agents to optimize energy trading strategies based on Q-value updates.
   - **PPO Implementation:** Trains agents to perform energy trading using policy optimization techniques.

3. **Auction-Based Energy Trading:**
   - Implements a market clearing mechanism for matching buyers and sellers in P2P energy markets.
   - Includes price optimization through a dedicated price advisor agent.

4. **Performance Metrics:**
   - Evaluation of energy cost reduction, peak hour demand minimization, and electricity revenue increase.
   - Comparison of DQN and PPO performance across different scenarios.

---

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/peer-to-peer-energy-trading
   cd peer-to-peer-energy-trading
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.8 or later installed. Then, install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset:**
   Place the energy demand and generation data files in the `datasets/` directory. Sample data for Ireland and Finland are included for convenience.

---

## Usage

To run the simulations, configure the desired parameters in the `configs/` directory and execute the corresponding script:

1. **Train DQN Agent:**
   ```bash
   python src/agents/train_dqn.py --config configs/dqn_config.yaml
   ```

2. **Train PPO Agent:**
   ```bash
   python src/agents/train_ppo.py --config configs/ppo_config.yaml
   ```

3. **Evaluate Performance:**
   ```bash
   python src/evaluate.py --config configs/evaluation_config.yaml
   ```

Results (e.g., energy costs, peak hour demand, and revenue metrics) will be stored in the `results/` directory.

---

## Results

The paper reports the following key findings:
- **Electricity Cost Reduction:**
  - DQN reduced electricity costs by 14.2% in Ireland and 5.16% in Finland.
- **Electricity Revenue Increase:**
  - DQN increased electricity revenue by 7.24% in Ireland and 12.73% in Finland.
- **Peak Hour Demand Reduction:**
  - PPO achieved the lowest peak hour demand, reducing it by 55.5% in Ireland.
  - DQN reduced peak hour demand by 50.0% in Ireland and 27.02% in Finland.

These results demonstrate the effectiveness of combining MARL algorithms with P2P trading mechanisms for decentralized energy management in rural communities.

---

## Future Work

Potential extensions and improvements include:
- Expanding the simulation to include more diverse datasets and energy profiles.
- Exploring additional MARL algorithms for energy optimization.
- Investigating the impact of external factors such as weather conditions on energy trading.

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use and modify the code for research and educational purposes. Proper citation to the original paper is required.

---

## Citation

If you use this code in your research, please cite the original paper:

```
@article{shah2023peer,
  title={Peer-to-Peer Energy Trading in Dairy Farms using Multi-Agent Reinforcement Learning},
  author={Mian Ibad Ali Shah, Marcos Eduardo Cruz Victorio, Maeve Duffy, Enda Barrett, Karl Mason},
  journal={arXiv preprint arXiv:2511.23148},
  year={2023}
}
```

---

## Contact

For questions, issues, or contributions, please open an issue on this repository or contact the authors via the email provided in the paper.