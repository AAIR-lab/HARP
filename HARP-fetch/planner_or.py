from openravepy import *
import numpy as np
import time
import sys
import os

envnum = sys.argv[1]
NUMRUNS = int(sys.argv[2])
MAXTIME = float(sys.argv[3])
planner_name = sys.argv[4]

# envnum = '10.0'
# NUMRUNS = 100
# MAXTIME = 5.0
# planner_name = 'OMPL_RRT'
visualize = False

env = Environment()
if visualize:
    env.SetViewer('qtcoin')

env.Load('envs3d/' + envnum + '.xml')
collisionChecker = RaveCreateCollisionChecker(env, 'fcl_') #fcl_,pqp,bullet
#collisionChecker.SetCollisionOptions(CollisionOptions.Distance|CollisionOptions.Contacts) # doesnt work with fcl
env.SetCollisionChecker(collisionChecker)

# load robots
robot = env.GetRobots()[0]

# 10.2.2, 10.0.2
if envnum in ['10.2', '10.0']:
    start = [0.5443226027230188, 0.9758539009870815, -1.7579191386448807, 0]
    goal = [-0.96970837, -0.82476306, -2.42265842, 0.5528569299999999]

# 8.0.3
if envnum == '8.0':
    start = [1.6, 1.0, 0, 0]
    goal = [-1.6, 1.0, 0, 0]

if robot.GetName() == 'robot1':
    robot.SetDOFVelocities([0.5,0.5,0.5])
    robot.SetActiveDOFs([0,1,2])
    robot.SetDOFLimits([-2.5, -2.5, -np.pi], [2.5, 2.5, np.pi])
elif robot.GetName() == 'lshaped':
    robot.SetDOFVelocities([0.5, 0.5, 0.5, 0.5, 0.5])
    robot.SetActiveDOFs([0,1,2,4])
    robot.SetDOFLimits([-2.5, -2.5, -np.pi, 0, -np.pi/2], [2.5, 2.5, np.pi, 0, np.pi/2])
    
robot.SetActiveDOFValues(start)
total_success = 0

if not os.path.isdir(os.path.join('results', 'test', 'env' + envnum + '.1')):
    os.makedirs(os.path.join('results', 'test', 'env' + envnum + '.1'))

results_file = open(os.path.join('results', 'test', 'env' + envnum + '.1', 'plan_' + planner_name.lower() + '.csv'), 'a')

for i in range(NUMRUNS):
    print("total_success, current run:", total_success, i)
    print("planner: ", planner_name)
    print("Time limit: ", MAXTIME)
    success = False
    with robot:
        planner = RaveCreatePlanner(env, planner_name)
        params = Planner.PlannerParameters()
        params.SetRobotActiveJoints(robot)
        params.SetConfigAccelerationLimit(robot.GetActiveDOFMaxAccel())
        params.SetConfigVelocityLimit(robot.GetActiveDOFMaxVel())
        params.SetInitialConfig(start)
        params.SetGoalConfig(goal)
        params.SetExtraParameters('<range>0.2</range>')
        params.SetExtraParameters('<time_limit>'+str(MAXTIME)+'</time_limit>')
        planner.InitPlan(robot, params)
        traj = RaveCreateTrajectory(env, '')
        with CollisionOptionsStateSaver(env.GetCollisionChecker(), CollisionOptions.ActiveDOFs):
            starttime = time.time()
            result = planner.PlanPath(traj)
            print(result)
            plantime = time.time() - starttime
    if not result == PlannerStatus.HasSolution:
        print 'FAILED'
        continue
    else:
        print("found in ", plantime)
        result = planningutils.RetimeTrajectory(traj)
        if not result == PlannerStatus.HasSolution:
            print 'FAILED'
            continue
        print 'PASSED'
        success = True
        if visualize:
            with robot:
                robot.GetController().SetPath(traj)
            robot.WaitForController(0)    
            print(traj.GetDuration())
    if success:
        total_success += 1

print(total_success)
results_file.write(str(MAXTIME) + ',' + str(total_success) + '\n')
results_file.close()