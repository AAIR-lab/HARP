import time
import os
from openravepy import *
import openravepy
import pickle
import sys
import numpy as np
# from prpy.planning import ompl, CBiRRTPlanner
sys.path.append('envs/3d/robots/fetch/')
import fetch
import math

class MotionPlansGenerator:
    
    def __init__(self, env, envnum, runnum, fetch_robot, motionplans_per_sample=50, motionplanning_time_limit=180.):
        self.env = env
        self.robot = env.GetRobots()[0]
        self.fetch_robot = fetch_robot
        # self.cdmodel = databases.convexdecomposition.ConvexDecompositionModel(self.robot)
        self.envnum = envnum
        self.runnum = runnum
        self.motionplans_per_sample = motionplans_per_sample
        self.single_path_max_retries = 1
        self.motionplanning_time_limit = motionplanning_time_limit
        self.OMPL = True
        # if not self.cdmodel.load():
            # self.cdmodel.autogenerate()

    def plan_to_configuration(self, start, goal):
        '''
        find motion plan to given IK
        goal: IK to move arm to
        '''
        planner = RaveCreatePlanner(self.env, 'OMPL_RRTConnect')
        params = Planner.PlannerParameters()
        params.SetRobotActiveJoints(self.robot)
        params.SetConfigAccelerationLimit(self.robot.GetActiveDOFMaxAccel())
        params.SetConfigVelocityLimit(self.robot.GetActiveDOFMaxVel())
        params.SetInitialConfig(start)
        params.SetGoalConfig(goal)
        params.SetExtraParameters('<range>0.2</range>')
        params.SetExtraParameters('<time_limit>'+str(self.motionplanning_time_limit)+'</time_limit>')
        planner.InitPlan(self.robot, params)
        traj = RaveCreateTrajectory(self.env, '')
        with CollisionOptionsStateSaver(self.env.GetCollisionChecker(), CollisionOptions.ActiveDOFs):
            starttime = time.time()
            result = planner.PlanPath(traj)
        if not result == PlannerStatus.HasSolution:
            return None
        result = planningutils.RetimeTrajectory(traj)
        if not result == PlannerStatus.HasSolution:
            return None
        return traj

    def plan_to_configuration_prpy(self, start, goal):
        # planner = ompl.OMPLPlanner('RRTConnect')
        planner = CBiRRTPlanner(timelimit=self.motionplanning_time_limit)
        simplifier = ompl.OMPLSimplifier()
        # Motion Planning to reach joint state value(s)
        # Get trajectory from planner based on type of goal config passed
        # ( config a.k.a ik solutions a.k.a joint states )
        try:
            trajectory_object = planner.PlanToConfigurations(self.robot, [goal])
            if hasattr(planner, 'default_ompl_args'):
                print("simplifying..")
                # If planner is from OMPL, then simplify the trajectory
                trajectory_object = simplifier.ShortcutPath(self.robot,trajectory_object)
        except Exception as e:
            print("Exception ", e)
            return None
        print("retiming..")
        # Retime and serialize the trajectory
        _ = planningutils.RetimeTrajectory(trajectory_object)
        # trajectory_object = trajectory_object.serialize()
        return trajectory_object

    def rotate_z(self, rot_angle):
        # rotate around z axis
        rotation_matrix = np.identity(4)
        rotation_matrix[0][0] = math.cos(rot_angle)
        rotation_matrix[0][1] = -(math.sin(rot_angle))
        rotation_matrix[1][0] = math.sin(rot_angle)
        rotation_matrix[1][1] = math.cos(rot_angle)
        return rotation_matrix

    def rotate_y(self, rot_angle):
        # rotate around y axis
        rotation_matrix = np.identity(4)
        rotation_matrix[0][0] = math.cos(rot_angle)
        rotation_matrix[0][2] = math.sin(rot_angle)
        rotation_matrix[2][0] = -math.sin(rot_angle)
        rotation_matrix[2][2] = math.cos(rot_angle)
        return rotation_matrix

    def rotate_x(self, rot_angle):
        # rotate around x axis
        rotation_matrix = np.identity(4)
        rotation_matrix[1][1] = math.cos(rot_angle)
        rotation_matrix[1][2] = -(math.sin(rot_angle))
        rotation_matrix[2][1] = math.sin(rot_angle)
        rotation_matrix[2][2] = math.cos(rot_angle)
        return rotation_matrix
        
    def translate(self, x, y, z):
        # rotate around x axis
        rotation_matrix = np.identity(4)
        rotation_matrix[0][3] = x
        rotation_matrix[1][3] = y
        rotation_matrix[2][3] = z
        return rotation_matrix

    def get_pickup_pose(self):
        pickup_poses = [
            [self.translate(-0.2,0,0.05)],
            [self.rotate_z(np.pi/2), self.translate(-0.2,0,0.05)],
            [self.rotate_z(np.pi), self.translate(-0.2,0,0.05)],
            [self.rotate_z(-np.pi/2), self.translate(-0.2,0,0.05)],

            [self.rotate_y(np.pi/2), self.translate(-0.27, 0, 0)],
            [self.rotate_y(np.pi/2), self.translate(-0.27, 0, 0), self.rotate_x(np.pi/2)],
            [self.rotate_y(np.pi/2), self.translate(-0.27, 0, 0), self.rotate_x(np.pi)],
            [self.rotate_y(np.pi/2), self.translate(-0.27, 0, 0), self.rotate_x(-np.pi/2)]
        ]
        possible_pickups = []
        # Get list of possible pickups
        for transformations in pickup_poses:
            grasp_pose = self.env.GetKinBody('pick_object').GetTransform()
            for transform in transformations:
                grasp_pose = np.matmul(grasp_pose, transform)
            ik_sols = self.fetch_robot.get_ik_solutions(grasp_pose, True)
            if len(ik_sols) > 0:
                idx = np.random.randint(len((ik_sols)))
                random_pickup_config = ik_sols[idx]
                possible_pickups.append(random_pickup_config)
        print("{total_grasps} possible grasps".format(total_grasps=len(possible_pickups)))
        if len(possible_pickups) == 0:
            return None
        else:
            # Randomly choose a pickup pose
            idx = np.random.randint(0, len((possible_pickups)))
            pickup_goal_config = possible_pickups[idx]
            return pickup_goal_config

    def get_goal(self):
        '''
        Return a random possible pickup dof configuration 
        '''
        part_model = env.ReadKinBodyURI('envs/3d/rectangle_bar.dae')
        env.AddKinBody(part_model)
        
        while True:
            # Place the object randomly on the table
            obj = self.env.GetKinBody('pick_object')
            table_wrt_world = self.env.GetKinBody('table61').GetTransform()
            x = (0.9)*np.random.random() + (-0.45)
            y = (0.9)*np.random.random() + (-0.45)
            t = self.translate(x, y, 0.14)
            obj.SetTransform(np.matmul(table_wrt_world, t))
            # If not in collision and pickup possible
            if not self.env.CheckCollision(obj):
                goal_config = self.get_pickup_pose()
                if goal_config is not None:
                    return goal_config

    def get_start(self):
        '''
        return first non collision dof configuration found
        '''
        llimits = self.robot.GetActiveDOFLimits()[0]
        ulimits = self.robot.GetActiveDOFLimits()[1]
        while True:
            start = []
            for i in range(len(ulimits)):
                dof_val = np.random.uniform(llimits[i], ulimits[i])
                start.append(dof_val)
                
            self.robot.SetActiveDOFValues(start)
            if not self.env.CheckCollision(self.robot) and not self.robot.CheckSelfCollision():
                break
        
        end_effector_transform = self.robot.GetManipulators()[0].GetTransform()
        ik_sols = self.fetch_robot.get_ik_solutions(end_effector_transform, True)
        return ik_sols[0]

    def generate(self):
        trajectory_info = []

        starttime = time.time()
        goal = self.get_goal()
        print("Goal: {goal}".format(goal=str(goal)))
        print("Goal generated in: {total_time}".format(total_time=(time.time() - starttime)))
        
        mp_i = 1
        while mp_i <= self.motionplans_per_sample:
            print("**********Currently processing: {env}.{runnum} #{mp_i}**********".format(env=self.envnum, runnum=self.runnum, mp_i=mp_i))
            
            starttime = time.time()
            start = self.get_start()
            print("Start: {start}".format(start=str(start)))
            print("Start generated in: {total_time}".format(total_time=(time.time() - starttime)))
    
            if self.env.GetViewer() is not None:
                self.robot.SetActiveDOFValues(goal)
                time.sleep(2)
                self.robot.SetActiveDOFValues(start)
                time.sleep(2)

            self.robot.SetActiveDOFValues(start)
            starttime = time.time()
            traj = self.plan_to_configuration(start, goal)
            print("Planning or completed in: {total_time}".format(total_time=(time.time() - starttime)))
            
            if traj is None: # If no MP found within timelimit, restart
                continue

            # Execute trajectory
            if self.env.GetViewer() is not None:
                with self.robot:
                    self.robot.GetController().SetPath(traj)
                self.robot.WaitForController(0)

            # Save points traversed in the trajectory
            totaldof = len(self.robot.GetActiveDOFValues())
            movepath = []
            sleeptime = 0.01
            starttime = time.time()
            while time.time()-starttime <= traj.GetDuration():
                curtime = time.time() - starttime
                trajdata = traj.Sample(curtime) 
                movepath.append(trajdata[:totaldof])
                time.sleep(sleeptime)
            
            # Add end location (Sampled trajectory might not have final end point)
            trajdata = traj.Sample(traj.GetDuration())
            movepath.append(trajdata[0:totaldof])
            print("Sampled path length: {length}".format(length=len(movepath)))
            if len(movepath) == 1:
                print("single point path. REJECT")
                continue
            trajectory_info.append({
                'start': start,
                'goal': goal,
                'traj': traj.serialize(),
                'path': movepath
            })
            mp_i += 1
            self.robot.WaitForController(0)

        traj_info_path = os.path.join(DATADIR, str(self.envnum) + '.' + str(self.runnum) + '_traj.pkl')
        pickle.dump(trajectory_info, open(traj_info_path, 'wb'))
        return


if __name__ == '__main__':

    envPath = sys.argv[1]
    envnum = sys.argv[2]
    runnum = int(sys.argv[3])
    visualize = False

    # envPath = 'envs/3d/4.0.dae'
    # envnum = '4.0'
    # runnum = 1
    # visualize = True

    DATADIR = os.path.join('data', 'env' + str(envnum), 'data')
    if not os.path.isdir(DATADIR):
        os.makedirs(DATADIR)

    env = Environment()
    env.Load(envPath)
    if visualize:
        env.SetViewer('qtcoin') 
    # set collision checker to Bullet (default collision checker might not recognize cylinder collision for Ubuntu) (causes issues when moving joints)
    collisionChecker = RaveCreateCollisionChecker(env, 'pqp')
    collisionChecker.SetCollisionOptions(CollisionOptions.Contacts)
    env.SetCollisionChecker(collisionChecker)
    
    initial_transform = pickle.load(open(os.path.join('envs','3d','fetch_transforms', str(envnum) + '.pkl'), 'rb'))
    fetch_robot = fetch.FetchRobot(env, initial_transform)

    generator = MotionPlansGenerator(env, envnum, runnum, fetch_robot)
    generator.generate()
