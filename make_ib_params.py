from params import *
from consts import *


betas = np.array([1,2,4])
alphas = np.array([0.01, 0.05])
mstars = np.array([1, 2.5])*MSUN
sigmafs = np.array([0.5, 1, 2])
mbhs = np.array([5e5, 1e6, 5e6])*MSUN

fs = {mbhs[0]: {sigmafs[0]: 1.07,   sigmafs[1]: 1.03, sigmafs[2]: 1.005},                                                      
      mbhs[1]: {sigmafs[0]: 1.11,   sigmafs[1]: 1.05, sigmafs[2]: 1.02},                                                       
      mbhs[2]: {sigmafs[0]: 1.28, sigmafs[1]: 1.16, sigmafs[2]: 1.1}}


#fs = {mbhs[0]: {sigmafs[0]: 3.77,   sigmafs[1]: 1.80, sigmafs[2]: 1.03},
#      mbhs[1]: {sigmafs[0]: 6.30,   sigmafs[1]: 2.30, sigmafs[2]: 1.20},
#      mbhs[2]: {sigmafs[0]: np.inf, sigmafs[1]: 19.4, sigmafs[2]: 2.19}}

pairings = [[beta, alpha, mstar, sigmaf, mbh] for beta in betas for alpha in alphas for mstar in mstars for sigmaf in sigmafs for mbh in mbhs]

#pairings = [[fbr0, beta, mbh, age, mstar] for fbr0 in fbr0s for beta in betas for mbh in mbhs for age in ages for mstar in mstars]

print(f"total number of pairings is {len(pairings)}")

m=30
parameters = []
j=0; a=0; b=0;k=0
for i, pairing in enumerate(pairings):
    dir=f"/global/scratch/users/dcarrel/ib_runs_{k:02d}"+f"/{j%m:02d}"
    f2u = fs[pairing[4]][pairing[3]]
    if np.isinf(f2u):
        continue
    param = Params(SIM_DIR=dir,
                     ALPHA=pairing[1], BE_CRIT=-0.1, NR=96, FWIND=1,
                     GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=1e-4,
                    TF=2*YEAR, PRINT=True,
                   BETA=pairing[0],
                   MBH=pairing[4],
                   MAGE=0.6,
                   MSTAR=pairing[2],
                   SIGMAF=f2u)
    if param.CAPTURE:
        a+=1
        if param.FALLBACK_FAILURE:
            b+=1
        continue
    elif param.FALLBACK_FAILURE:
        b+=1
        continue
    else:
        param.save()
        j+=1
        if j % m == 0:
            k+=1
print(f"total params saved is {j}")
print(f"some issue with fallback {b}")
print(f"some issue with capture radius {a}")
