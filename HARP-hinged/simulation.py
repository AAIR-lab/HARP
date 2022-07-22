import pdb
import time
from sys import *
from generate_mp import MotionPlansGenerator
from generate_labels import LabelGenerator
from openravepy import *
from numpy import *
import os


if len(argv) == 5:
    envPath = argv[1]
    envnum = float(argv[2])
    runnum = int(argv[3])
    mode = argv[4]
else:
    print '\nincorrect number of arguments passed (4 required)'
    exit(0)

# envPath = 'envs3d/5.2.xml'
# envnum = 5.2
# runnum = 1
# mode = 'motion-planning'

try:
    env = Environment()
    env.Load(envPath)
    
    # env.SetViewer('qtcoin')
    
    # set collision checker to Bullet (default collision checker might not recognize cylinder collision for Ubuntu) (causes issues when moving joints)
    collisionChecker = RaveCreateCollisionChecker(env, 'fcl_')
    env.SetCollisionChecker(collisionChecker)

    # load robots
    robot = env.GetRobots()[0]
    env.UpdatePublishedBodies()
    time.sleep(0.1)

    bounds = array([[-2.5, -2.5, -3.14], [2.5, 2.5, 3.14]]) # all random envs are built using the same base world, thus they have the same world bounds
    envmin = array([-2.5, -2.5, -2.5])
    envmax = array([2.5, 2.5, 2.5])

    # Create directories if not exists
    envdir = os.path.join('data', 'env' + str(envnum))
    if not os.path.isdir(envdir):
        os.makedirs(envdir)
    datadir = os.path.join(envdir, 'data')
    if not os.path.isdir(datadir):
        os.makedirs(datadir)
    inp = os.path.join(envdir, 'inp')
    if not os.path.isdir(inp):
        os.makedirs(inp)
    lbldir = os.path.join(envdir, 'lbl')
    if not os.path.isdir(lbldir):
        os.makedirs(lbldir)

    if mode == 'motion-planning':
        # perform motion planning
        generator = MotionPlansGenerator(robot, envnum, bounds, [envmin, envmax], runnum)
        starttime = time.time()
        generator.generate()
        print('timing: '+str(time.time()-starttime))
    elif mode == 'generate-label':
        generator = LabelGenerator(robot, envnum, bounds, [envmin, envmax], runnum)
        starttime = time.time()
        generator.generate()
        print('timing: '+str(time.time()-starttime))
    else:
        print("no matching mode for {mode}".format(mode=mode))
    
finally:
    # destory environment
    RaveDestroy()

