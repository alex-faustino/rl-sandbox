from Double_InvertedPend_v0 import doublePendulum
import numpy as np
import matplotlib.pyplot as plt

env = doublePendulum()
state_rec = []
action_rec = []
steps = 100

for i in range(steps):
    action = env.action_space_sample()
    s,r = env.step(action)
    state_rec.append(s)
    action_rec.append(action)
    #env._render()

state_rec = np.reshape(state_rec, (steps,4))

plt.figure(2)
plt.subplot(411)
plt.plot(state_rec[:,0])
plt.ylabel('Angle Arm-1')
plt.subplot(412)
plt.plot(state_rec[:,1])
plt.ylabel('Rate Arm-1')
plt.subplot(413)
plt.plot(state_rec[:,2])
plt.ylabel('Angle Arm-2')
plt.subplot(414)
plt.plot(state_rec[:,3])
plt.ylabel('Rate Arm-2')
plt.xlabel('Time')
plt.show()

plt.figure(3)
plt.plot(action_rec)
plt.xlabel('Time')
plt.ylabel('Action')
plt.show()
