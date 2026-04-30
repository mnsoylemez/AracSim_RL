AracSim_RL — Asymmetric Adversarial RL for Autonomous Vehicle Navigation
Trained model checkpoints for the paper "Asymmetric Adversarial Reinforcement Learning for Robust Autonomous Vehicle Navigation". This repository contains the final trained weights of:

PPO agent (Araç) — drives the simulated vehicle
SAC adversary (Trafik Canavarı / Traffic Monster) — selects road difficulty parameters to challenge the agent
The agents were co-trained for 10,000 generations under PAIRED reward (Dennis et al., 2020). At convergence, the agent achieves 100% success rate on roads chosen by the trained adversary in evaluation.

Repository contents
checkpoints/
├── arac_latest.pt        ← Trained PPO agent (final policy)
├── tc_latest.pt          ← Trained SAC traffic monster (final policy)
└── arac_bc_anchor.pt     ← Behavioral cloning anchor (initial weak baseline)
play.py                   ← Standalone demo runner
README.md                 ← This file
Note: This repository contains only the trained checkpoints and demo code. The full training codebase (environment, training loops, etc.) is hosted separately. See Setup below.

Setup
1. Requirements
Python 3.10 or newer
PyTorch 2.0+ (CUDA optional)
pygame 2.6+
numpy
pip install torch pygame numpy
2. Install the codebase
The training and environment code is in a separate repository (aracsim_rl_v2). Place this folder alongside the codebase, or copy the checkpoints into the codebase's checkpoints/ directory.

aracsim_rl_v2/
├── agents/
├── config.py
├── data/
├── egitim/
├── env/
├── physics/
├── render/
├── road/
├── sim/
├── checkpoints/        ← put .pt files here
│   ├── arac_latest.pt
│   ├── tc_latest.pt
│   └── arac_bc_anchor.pt
├── play.py
└── README.md
3. Run the demo
python play.py
A pygame window opens. The trained PPO agent drives the vehicle while the trained SAC adversary selects new road parameters at the start of each episode.

Command-line options
Flag	Effect
(none)	Adversary chooses difficulty (default — most interesting)
--easy	Fixed easy preset (curve=0.18, μ=0.90, width=9.0 m)
--default	Fixed medium preset (curve=0.35, μ=0.55, width=7.5 m)
--hard	Fixed hard preset (curve=0.65, μ=0.20, width=5.5 m)
--episodes N	Stop after N episodes (default: unlimited)
Keyboard during play
Key	Action
R	Restart the current scenario
N	Start a new randomly chosen scenario
ESC	Quit
What you should see
Pseudo-3D pygame visualization of a curved road with friction-dependent surface coloring (dry asphalt, wet, snow, dirt)
HUD showing the current adversary parameters, episode result, and rolling success rate
The agent completes ~95–100% of episodes, including roads with μ = 0.10 (extreme low friction) and curve amplitude up to 0.80
Occasional crashes — the adversary still finds genuine challenges (paired = +1 events)
After ~30 demo episodes the success counter should be at or above 90%.

Citation
If you use these checkpoints or the methodology in your research, please cite the accompanying paper.

@article{aracsim_rl_2026,
  title  = {Asymmetric Adversarial Reinforcement Learning for Robust
            Autonomous Vehicle Navigation},
  author = {Söylemez, M. N.},
  year   = {2026}
}
License
Model checkpoints are released for research and educational use.

