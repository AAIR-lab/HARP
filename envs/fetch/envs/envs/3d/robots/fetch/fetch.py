from openravepy import *
import numpy as np
import multiprocessing
import os

class FetchRobot(object):
    def __init__(self, env, initial_transform=None):
        self.env = env
        self.jointnames = (["torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint", 
                            "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"])
        self.active_manipulators = {'arm_and_torso': 'arm_torso'}
        self.armTuckDOFs = [0.00000000e+00, 1.71996798e+00, 2.70622649e-06,
                            0.00000000e+00, 0.00000000e+00, 4.83987850e-16,
                            0.00000000e+00, 7.51135265e-16, 0.00000000e+00,
                            1.39998343e+00, 1.32000492e+00, 0.00000000e+00,
                            -1.99845334e-01, 1.66000647e+00, -1.34775426e-06]

        self.URDF_DIR = os.path.abspath(os.path.dirname(__file__))+'/'
        self.fetch_urdf = self.URDF_DIR + '/URDF/fetch.urdf' 
        self.fetch_srdf = self.URDF_DIR + '/URDF/fetch.srdf'
        module = RaveCreateModule(self.env, 'urdf')
        with self.env:
            self.set_paths()
            name = module.SendCommand('loadURI ' + self.fetch_urdf + ' ' + self.fetch_srdf)
            self.reset_paths()

        body = self.env.GetKinBody(name)
        self.robot = self.env.GetRobot(name)
        self.robot.SetActiveDOFs([self.robot.GetJoint(name).GetDOFIndex() for name in self.jointnames])
        self.robot.SetActiveManipulator(self.active_manipulators['arm_and_torso'])
        self.initGripper()
        self.setJointAcclerationLimits(2)
        self.robot.SetAffineTranslationMaxVels([10.5, 10.5, 10.5])
        self.robot.SetAffineRotationAxisMaxVels(np.ones(4))
        if initial_transform is None:
            self.robot.SetTransform(np.eye(4))
        else:
            self.robot.SetTransform(initial_transform)

        self.env.UpdatePublishedBodies()
        self.robot.SetDOFValues(np.asarray(self.armTuckDOFs))
        self.openGrippers()

    def initGripper(self):
        """Setup gripper closing direction and tool direction """
        gripperManip = self.robot.GetActiveManipulator()
        gripperIndices = gripperManip.GetGripperIndices()
        closingDirection = np.zeros(len(gripperIndices))
        for i, gi in enumerate(gripperIndices):
            closingDirection[i] = -1.
        gripperManip.SetChuckingDirection(closingDirection)
        gripperManip.SetLocalToolDirection([1, 0, 0])

    def setJointAcclerationLimits(self, val):
        accel_limits = self.robot.GetDOFAccelerationLimits()
        manipulator = self.robot.GetActiveManipulator()
        accel_limits[manipulator.GetArmIndices()] = [val] * manipulator.GetArmDOF()
        self.robot.SetDOFAccelerationLimits(accel_limits)

    def mapPhysicalRobotJointValues(self):
        import pickle
        joint_dump = open(Config.FETCH_INIT_STATE_PICKLE, "rb")
        joint_poses = pickle.load(joint_dump)
        joint_dump.close()
        robot_joints = ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint',         
                        'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
        vals = [2,6,7,8,9,10,11,12]
        self.robot.SetActiveDOFs([self.robot.GetJoint(joint).GetDOFIndex() for joint in robot_joints])
        self.robot.SetActiveDOFValues([v for i,v in enumerate(joint_poses[0]) if i in vals])

    def openGrippers(self):
        taskmanip = interfaces.TaskManipulation(self.robot)
        with self.robot:
            taskmanip.ReleaseFingers()
        self.robot.WaitForController(0)

    def get_ik_solutions(self,end_effector_solution, check_collisions=False):
        if check_collisions:
            filter_option = IkFilterOptions.CheckEnvCollisions
        else:
            filter_option = IkFilterOptions.IgnoreEndEffectorCollisions
        with self.env:
            ikmodel = databases.inversekinematics.InverseKinematicsModel(self.robot, iktype=IkParameterization.Type.Transform6D)
            if not ikmodel.load():
                ikmodel.autogenerate()
            solutions = ikmodel.manip.FindIKSolutions(end_effector_solution, filter_option)
        if len(solutions) == 0:
            print "NO IKs found, Probably Un-reachable transform"
        return solutions

    def set_paths(self):
        # Adds correct path in URDF files
        sedstr = "sed -i \"s|project_directory|"+self.URDF_DIR[:-1]+"|g\" "+self.URDF_DIR
        os.system(sedstr +"/URDF/fetch.urdf")
 
    def reset_paths(self):
        sedstr = "sed -i \"s|"+self.URDF_DIR[:-1]+"|project_directory|g\" "+self.URDF_DIR
        os.system(sedstr +"/URDF/fetch.urdf")