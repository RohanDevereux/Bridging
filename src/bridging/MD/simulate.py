from pathlib import Path

from openmm import Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmm.app import Simulation, PDBFile
from openmm.app import PME, HBonds
from openmm.app import StateDataReporter
from openmm.unit import kelvin, picosecond, femtoseconds, atmosphere, nanometer

from mdtraj.reporters import HDF5Reporter

from .config import (
    TIME_STEP_FS, FRICTION_PER_PS, PRESSURE_ATM,
    MINIMIZE_MAX_ITERS, EQUIL_STEPS, PROD_STEPS, REPORT_EVERY_STEPS
)
from .save_utils import get_ca_atom_indices, write_ca_topology_pdb

def run_simulation(forcefield, modeller, out_dir, temperature_k):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
        rigidWater=True,
        ewaldErrorTolerance=1e-4,
    )
    system.addForce(MonteCarloBarostat(PRESSURE_ATM * atmosphere, temperature_k * kelvin))

    integrator = LangevinMiddleIntegrator(
        temperature_k * kelvin,
        FRICTION_PER_PS / picosecond,
        TIME_STEP_FS * femtoseconds,
    )

    platform = Platform.getPlatformByName("CUDA")
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    simulation.minimizeEnergy(maxIterations=MINIMIZE_MAX_ITERS)
    simulation.context.setVelocitiesToTemperature(temperature_k * kelvin)

    ca_idx = get_ca_atom_indices(modeller.topology)

    simulation.reporters.append(HDF5Reporter(
        str(out_dir / "traj_ca.h5"),
        REPORT_EVERY_STEPS,
        atomSubset=ca_idx,
    ))
    simulation.reporters.append(StateDataReporter(
        str(out_dir / "log.txt"),
        REPORT_EVERY_STEPS,
        step=True,
        time=True,
        potentialEnergy=True,
        temperature=True,
        speed=True,
    ))

    write_ca_topology_pdb(out_dir / "topology_ca.pdb", modeller.topology, modeller.positions)

    simulation.step(EQUIL_STEPS)
    simulation.step(PROD_STEPS)

    state = simulation.context.getState(getPositions=True)
    with open(out_dir / "final.pdb", "w") as f:
        PDBFile.writeFile(simulation.topology, state.getPositions(), f)

    return {"traj_ca_h5": "traj_ca.h5", "topology_ca_pdb": "topology_ca.pdb"}
