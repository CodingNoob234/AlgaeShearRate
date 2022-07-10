Project Title: Simulating cyanobacteria in a simple shear flow in the Stokes regime
Author: Joey Besseling
Bachelor Thesis Physics & Astronomy at University of Amsterdam

### Motivation and Theory
The goal of my bachelor thesis is simulating a small number of cyanobacteria in a simple shear flow and observe the break up time against different varied parameters. The cyanobacteria are approached as rigid spheres, connected through bending springs.  A break up is considered as one of the springs being extended past a certain threshold. The flow is considered a low Reynolds number flow, therefore the Stokes' equations can be used. Faxen's law is an extension to Stokes' law, taking into account particle-particle interactions. The trajectories are computed using Runge-Kutta 4th-order method. In each iteration the positions are updated, an if-statement checks if any of the springs is extended past a break up condition. If so, the simulation is ended or the connection is removed after which the simulation continues.

### Applications
The main functions are:
- simulate one system, plot or animate, and register a break up
- vary the shear rate for a system and return the plot 'shear-rate vs. break-up time'
- vary the initial distance for a system and return the plot 'shear-rate vs. break up time'
- find the minimum shear rate for a given system up to a specified accuracy, using bisection method
- find the minimum shear rate for multiple systems and return the plot 'type of system vs. minimum shear rate'

### Structure of script
- 'class Particle' for holding the properties of the particles
- functions for initializing, simulating and repeating simulations for varied parameters
- functions for plotting, animating and making gifs
- input requests for parameters and type of simulations to perform, and executing the requested simulations

### Usage
- Scroll through to line 836, where all main functions are executed.
- When the script is run, a number of input paramteres are requested depending on the chosen simulation, like what parameters are varied and for what type of systems.
- Not all parameters are requested through the console; break up threshold, simulation time, initial distance and others can be changed manually from lines 824 to 834
