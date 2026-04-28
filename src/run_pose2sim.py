import os
os.chdir('data/pose2sim_demo')

from Pose2Sim import Pose2Sim


print("=" * 50)
print("STEP 1: Camera Calibration")
print("=" * 50)
Pose2Sim.calibration()

print("=" * 50)
print("STEP 2: 2D Pose Estimation")
print("=" * 50)
Pose2Sim.poseEstimation()

print("=" * 50)
print("STEP 3: Person Association")
print("=" * 50)
Pose2Sim.personAssociation()

print("=" * 50)
print("STEP 4: Triangulation")
print("=" * 50)
Pose2Sim.triangulation()

print("=" * 50)
print("STEP 5: Filtering")
print("=" * 50)
Pose2Sim.filtering()

print("=" * 50)
print("DONE! Check data/pose2sim_demo/ for output files")
print("=" * 50)
