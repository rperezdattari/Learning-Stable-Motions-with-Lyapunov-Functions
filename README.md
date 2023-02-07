## Learning Control Lyapunov functions to Ensure Stable Motions

This code implements the paper "Learning control Lyapunov function to ensure stability of dynamical system-based robot reaching motions" in python. For details, please refer to:
```
   S.M. Khansari-Zadeh and A. Billard (2014), "Learning Control Lyapunov Function
   to Ensure Stability of Dynamical System-based Robot Reaching Motions."
   Robotics and Autonomous Systems, vol. 62, num 6, p. 752-765.
```

The original MATLAB implementation of this paper can be found at:

https://bitbucket.org/khansari/clfdm/.

This code started as a branch of:

https://github.com/robotsorcerer/LyapunovLearner.

Differently from this other repository, here we only focus on the method implementation. Moreover, the [LASA handwriting dataset](https://bitbucket.org/khansari/lasahandwritingdataset/src/master/) has been included for testing.

## Examples
### Energy levels Lyapunov function
- **Blue**: demonstrations
- **Red**: trajectories from learned dynamical system

<img src="/src/images/example_energy_levels.png" width="500" />

### Vector field learned dynamical system
- **White**: demonstrations
- **Red**: trajectories from learned dynamical system

<img src="/src/images/example_vector_field.png" width="500" />

## Setup
Install as follows:
```
  pip install -r requirements.txt
```

## Usage
In the folder `src` run:
```
  python main.py
```

To change parameters or select a different shape modify the file `config.py`. 

## Credits
- Olalekan Ogunmolu
- Rodrigo Pérez Dattari
- Rachel Skye Thompson
