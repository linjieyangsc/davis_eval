import os
import yaml
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
methods=['OSVOS_parent','MSK_parent','OSVOS_M','MSK_M','OSVOS','MSK','OSMN']
labels=['OSVOS-B','MaskTrack-B','OSVOS-M','MaskTrack-M','OSVOS','MaskTrack','Ours']
colors =['b:', 'g:', 'b--', 'g--', 'b', 'g', 'r']
for i,m in enumerate(methods):
    print 'loading ', m
    all_res= []
    max_len = 0
    res = yaml.load(open(m+'.yaml'))
    print res.keys()
    print type(res['sequence'])
    for k,seq in res['sequence'].iteritems():
        all_res.extend(seq['J']['raw'])
        max_len = max(max_len, len(seq['J']['raw'][0]))
    interp_res = []
    interp_len = 100
    max_len -= 2
    for seq in all_res:
        interp_seq = np.zeros((interp_len)) 
        seq = np.array(seq[1:-1])
        xp = np.arange(seq.size)
        yp = np.linspace(0,seq.size,interp_len) 

        interp_seq = np.interp(yp,xp,seq)
        interp_res.append(interp_seq)
    interp_res = np.array(interp_res)
    j_over_frame = np.mean(interp_res, axis=0)
    # remove the first and the last frame
    x = np.linspace(0, 1, interp_len)
    ax.plot(x, j_over_frame, colors[i], label=labels[i])
legend = ax.legend(loc='upper right', shadow=False)
ax.grid(True, linestyle='--')
ax.set_xlabel('ratio of sequence')
ax.set_xlim([0,1])
ax.set_ylabel('J mean')
plt.savefig('j_over_time.png',dpi=400)

