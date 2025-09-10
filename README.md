# 1D Blood Flow Solver – Navier-Stokes Equations

This project implements a 1D numerical solver for blood flow in compliant arteries using the Navier-Stokes equations. It models wave propagation in elastic vessels and can be extended to networks of arteries with bifurcation junctions and Windkessel-type boundary conditions.

---

## Features

- 1D Navier-Stokes equations in conservative form
- Elastic wall models: linear and nonlinear pressure-area laws
- Discretization using Richtmyer (two-step Lax-Wendroff) scheme
- Supports bifurcating vessels with appropriate boundary conditions
- Custom vessel geometries and simulation parameters via `.ini` file
- Inflow and pressure signal as inlet boundary conditions
- Three-element Windkessel model for outlet boundary conditions
- Optional visualization given by `plot_results.py` script

---

## Installation

Requires Python 3 or higher. Recommended to use a virtual environment.

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage 

1. Edit the configuration file to define vessel geometry, parameters and simulation settings. Examples of configuration files in different vessel networks are given in `config.ini` and `config_bif.ini`.

2. To run the solver, open a terminal window in the folder where the python script is located and type the following command :

```bash
python3 Solver.py config.ini
```

3. The output files consists of both 1D and 3D graphs which are saved in data directory. The path where the files are saved are being printed by the program at the end of the simulation.

---

## Configuration File

The parameters are defined via an `.ini` file structured in three sections:
- `[Vessel]`: Corresponds to the vessel geometry and the pressure model, ru correspond to the radius at the inlet of the vessel and rd the radius at the outlet.
- `[Simulation]`: Space and time discretization of the simulation, choice of the inlet boundary condition with the possibility to either generate data or pick data you have.
- `[Parameters]`: Blood property, Three element Windkessel and elastances parameters
- `[Network]`: Connectivity table for the arteries in the network

---

## Examples

Two configuration files are provided to test the solver. To run a simulation with a single vessel and a non-linear pressure model:

```bash
python3 Artery.py config.ini
```

To run the simulation in a tiny vessel network consisting of one parent vessel and two daughter vessels with a linear pressure model:

```bash
python3 Artery.py config_bif.ini
```

---

## Author

This solver was developped by Victor Juille as part of a research internship on cerebral blood flow modeling, under the supervision of Pr Sampsa Pursiainen and Dr Maryamolsadat Samavaki at Tampere University.

**Contacts** : 
- victor.juille@etudiant.univ-reims.fr
- maryamolsadat.samavaki@uef.fi | m.samavaki@yahoo.com
- sampsa.pursiainen@tuni.fi

---

## Licence

This solver is licensed under the GNU GPLv3. See [LICENSE](https://github.com/victor-jle/Blood_flow/blob/main/LICENSE) for details.

---

