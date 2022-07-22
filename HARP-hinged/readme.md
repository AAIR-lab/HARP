## Data Generation

Generate training data and labels for a single environment  
`python simulation.py <path to .xml/.dae> <env_num> <run_num> motion-planning`  
`python simulation.py <path to .xml/.dae> <env_num> <run_num> generate-label`  

**Args**  
- `<path to .xml>`: path to your environment file e.g. envs3d/17.0.xml  
- `<env_num>`: name of the environment e.g. 17.0  
- `<run_num>`: suffix appended to your training sample

**Outputs**
If `run_num=1` and `env_num=17.0` the outputs will be  
- `data/17.0.1_traj.pkl`: list of all 50 mp trajectories in the sample. Each element in the list is a dictionary 
    ```
    {
        'start': [dof values],
        'goal': [dof values],
        'traj': traj.serialize(), # openrave trajectory
        'path': cleaned_movepath # sampled openrave trajectory
    }
    ```
- `data/17.0.1_chnls.npy`: raw_channel values from converting motion plans from env to pixels
- `inp/17.0.1.npy`: environment + goal condition
- `lbl/17.0.1.npy`: output gaze(critical regions) + dof values

Once all training data is generated run following scripts:  

1. `generate_augmentations.py`: Generate more samples by rotating and mirroring the samples  
2. `movedata.py`: Move all data to `segnet.tf-master/src/input/raw/`  

**Helper Scripts**  
`view_data.py`: Visualize generated data sample and augmented samples in matplotlib  
`view_traj.py`: Visualize running all trajectories of a data sample in the environment  
`./dataautomation3d.sh`: Automate generation of training data for an environments   

## Network Training  
Adapted from the codebase @[andreaazzini/segnet.tf](https://github.com/andreaazzini/segnet.tf)  

1. `tfrecorder.py`: Generate tf records for train and test set  
2. `train.py`: Train the network  

## Test Network

1. TF Records should have been generated from `tfrecorder.py`. Use one test image per run.
2. `test.py`: Generates the predictions for test records and creates a `pred.npy` and `samples.pkl` file under `results/test/env<envnum>/`

`samples.pkl`: Contains a list of all predicted critical regions and their orientation predicted bins e.g. `[minx, miny, maxx, maxy, [[rotation_min, rotation_max], .. ]]`

## Run LLP/LL-RM Planners

1. `ll.py`: Runs the LLP/LL-RM planners using the predicted samples from the network


### LLP/LL-RM readme
The LL planners requires an OpenRave environment, collision checker, and robot.

IMPORTANT: In ll.py, set MAXTIME allowed per plan on line 22, and set the mode ("llp" or "ll-rm") on line 23.
If importing the planner like in the example below, changing envnum (line 20) and NUMRUNS (lines 21) won't do anything.

This code is only set to work for 3 DOF or 10 DOF problems only. If you want to change this, you'll need to modify the compound_step() function on line 92.

If you want to take finer steps in you plan, lower the 3rd paramenter of self.step_from_to() in the compound_step() function on line 92.

Example usage: 

```
from ll import *

'''
env : loaded openrave environment object with robot added

jointlimits :   jointlimits[0] list of min limits for the joints of the robot and 
                (x_min,y_min,theta_min) ---> [J0_min,...,JN_min,x_min,y_min,theta_min]
                jointlimits[1] list of max limits for the joints of the robot and
                (x_max,y_max,theta_max) ---> [J0_max,...,JN_max,x_max,y_max,theta_max]

samples : list of the 2D critical configurations ----> [[x0,y0],...,[xN,yN]]

n : number of critical region verteces to include in the roadmap

m : number of uniformly sampled verteces to inlcude in the roadmap, this is 0 in LLP mode

start : list containing the start cofiguration of the plan ----> [J0_init,...,JN_init,x_init,y_init,theta_init]

goal : list containing the goal cofiguration of the plan ----> [J0_goal,...,JN_goal,x_goal,y_goal,theta_goal]

returns: tuple where [0] is a boolean indicating success or failure, and [1] is a list of configurations 
        connecting start and goal
'''

### For LLP mode

env = Environment()
env.Load("env.xml") 
jointlimits = [[-2.5, -2.5, -pi], [2.5, 2.5, pi]]
samples = [[0.25, 0.25], [0.75, 0.75]]
n = int(0.05*len(samples))
m = 0
start = [0,0,0]
goal = [1,1,0]
problem = LL(env, jointlimits, samples, n, m, start, goal)
success, path = problem.plan(start, goal)

### For LLRM mode 

env = Environment()
env.Load("env.xml") 
jointlimits = [[-2.5, -2.5, -pi], [2.5, 2.5, pi]]
samples = [[0.25, 0.25], [0.75, 0.75]]
n = int(0.05*len(samples))
m = int(n/10)
start = [0,0,0]
goal = [1,1,0]
problem = LL(env,jointlimits,samples,n,m)
success, path = problem.plan(start,goal)
```