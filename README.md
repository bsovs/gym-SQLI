# gym-SQLI

Simulation environment to test SQL Injection based attacks on a system using ML models.

Extension to the research project [CTF-SQL](https://github.com/FMZennaro/CTF-SQL) that implements the model and
simulations from the 2021 research
paper [Simulating SQL Injection Vulnerability Exploitation Using Q-Learning Reinforcement Learning Agents](https://arxiv.org/pdf/2101.03118.pdf)

### Requirements

The following code requires *numpy* and [opanai gym](https://www.gymlibrary.dev/).

### Installation

Clone this git and run `pip install -e gym-SQLI` to make this environment available to *OpenAI gym*.

### Content

Error Message Env [run test](./tests/backtest_error.py)

- Run model on simulation env over an action space to find SQL error messages
- [gym env](./sqli_sim/envs/error_env.py)

Flag Env (wip) [run test](./tests/backtest_error_flag.py)

- Run model on simulation env to use error messages (observations) to find a flag via SQLI
- PPO and DQN models available
- [gym env](./sqli_sim/envs/error_flag_env.py)
