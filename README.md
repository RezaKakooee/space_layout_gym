# Source Codes for RLDesigner

## Note:
Please open the `First_Paper` branch to access to the code!

## About RLDesigner

RLDesigner is a new reinforcement learning (RL) environment where agents are trained to address the general-porpuse problem of spatial layout planing (SLP). By proposing a new space partitioning approach, we formulize the SLP as Markov Decision Process (MDP) by which we cane use RL algorithms for designing layouts.

The RLDesigner is an OpenAI Gym compatible environment that allows easy use of different RL, particularly Deep RL algorithms. Besides, it is a customizable environment that enables the alteration of different parameters, architectural constraints, and geometrical and topological objectives in order to compose new design scenarios.

We publicly share the environment to encourage both RL and ARCH communities to use it for testing different RL algorithms or in their design practice.

## About this repository

The repository consists of two main directories. The fist one is `gym-floorplan` in which there are all scripts that developes the SLP environment. And the second one is `agens-floorplan` that includes all scrips that uses `RLlib` for training PPO algorithm.

The RLlib pipeline that we implemented almost contains a diverse set of modules one might need to train an RL agent. We refer the readers to the codes for more information.

There is also a simple scrip in directory `sb_agent` that shows how one can use `StableBaselines` with `RLDesigner` environment.

## Setup the environment

Like other Gym custom environment, the RLDesigner environment can be easily installed with `pip install -e gym-floorplan` command. Make sure to install the `requirements` too.

## Note:
This repository is only the public version of underdeveloping RLDesigner environment which is currently a private repository but will be public in the near future. As soon as we make it public, the new link will be shared here! In case you need access to the latest version, please feel free to contact the first author. 

## Citing RlDesigner

```bibtex
@article{kakooee2022rldesigner,
  title={RLDesigner: Toward Framing Spatial Layout Planning as a Markov Decision Process},
  author={Kakooee, Reza and Dillenburger, Benjamin},
  journal={The 15th European Workshop on Reinforcement Learning (EWRL 2022)},
  year={2022}
}
```
