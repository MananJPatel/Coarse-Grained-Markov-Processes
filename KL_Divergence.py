import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import norm
from scipy.integrate import quad

def p(x):
    return norm.pdf(x, 0, 3)

def q(x):
    return norm.pdf(x, 3, 3)

def KL(x):
    return p(x) * np.log( p(x) / q(x) )

range = np.arange(-20, 20, 0.002)
KLD, error = quad(KL, -20, 20) 
print( 'KL Divergence', KLD )
fig = plt.figure(figsize=(20, 10), dpi=90)

ax = fig.add_subplot(1,2,1) # ploting p(x) and q(x)
ax.grid(True)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position('zero')
ax.set_xlim(-15,15)
ax.set_ylim(-0.15,0.20)
ax.text(-2.5, 0.17, 'p(x)', horizontalalignment='center',fontsize=14)
ax.text(4.5, 0.17, 'q(x)', horizontalalignment='center',fontsize=14)
plt.plot(range, p(range))
plt.plot(range, q(range))

ax = fig.add_subplot(1,2,2) # ploting KL(x)
ax.grid(True)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position('zero')
ax.set_xlim(-15,15)
ax.set_ylim(-0.15,0.20)
ax.text(3.5, 0.17, r'$DK_{KL}(p|q)$', horizontalalignment='center',fontsize=14)
ax.plot(range, KL(range))
ax.fill_between(range, 0, KL(range))

plt.savefig('KullbackLeibler.png',bbox_inches='tight')
plt.show()