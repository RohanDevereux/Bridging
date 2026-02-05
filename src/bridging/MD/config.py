# Minimal, reasonable defaults.
# Tune on Sonic once you know what runtime/storage you can afford.

FORCEFIELD_FILES = ("amber14-all.xml", "amber14/tip3pfb.xml")

WATER_PADDING_NM = 1.0          # nm of solvent padding
IONIC_STRENGTH_M = 0.15         # molar

TIME_STEP_FS = 2.0              # fs
FRICTION_PER_PS = 1.0           # 1/ps
PRESSURE_ATM = 1.0              # 1 atm

MINIMIZE_MAX_ITERS = 1000
EQUIL_STEPS = 250_000           # 0.5 ns @ 2 fs
PROD_STEPS  = 2_500_000         # 5 ns @ 2 fs

REPORT_EVERY_STEPS = 5_000      # 10 ps @ 2 fs
