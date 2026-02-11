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

def _get_platform():
    for name in ["CUDA", "OpenCL", "CPU"]:
        try:
            return Platform.getPlatformByName(name)
        except Exception:
            continue
    raise RuntimeError("No OpenMM platform found.")


def _run_stage(simulation, total_steps, report_every, label):
    chunk = max(report_every * 10, report_every)
    completed = 0
    while completed < total_steps:
        step = min(chunk, total_steps - completed)
        simulation.step(step)
        completed += step
        print(f"[{label}] step {completed}/{total_steps}")


def _disulfide_residue_set(topology):
    disulfide = set()
    for a1, a2 in topology.bonds():
        if (
            a1.name == "SG"
            and a2.name == "SG"
            and a1.residue.name == "CYS"
            and a2.residue.name == "CYS"
        ):
            disulfide.add(a1.residue)
            disulfide.add(a2.residue)
    return disulfide


def _create_system(forcefield, modeller, ignore_external_bonds, residue_templates=None):
    return forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
        rigidWater=True,
        ewaldErrorTolerance=1e-4,
        ignoreExternalBonds=bool(ignore_external_bonds),
        residueTemplates=residue_templates or {},
    )


def build_system(forcefield, modeller, allow_ignore_external_bonds=False):
    residue_templates = {}
    residues = list(modeller.topology.residues())
    by_index = {res.index: res for res in residues}
    disulfide = _disulfide_residue_set(modeller.topology)

    for _ in range(25):
        try:
            return _create_system(
                forcefield,
                modeller,
                ignore_external_bonds=False,
                residue_templates=residue_templates,
            )
        except Exception as exc:
            msg = str(exc)
            if "Multiple non-identical matching templates found for residue" not in msg:
                if allow_ignore_external_bonds:
                    return _create_system(
                        forcefield,
                        modeller,
                        ignore_external_bonds=True,
                        residue_templates=residue_templates,
                    )
                raise

            parts = msg.split(":")
            if not parts:
                raise
            left = parts[0]
            right = parts[-1]
            try:
                res_idx = int(left.split("residue")[1].split("(")[0].strip())
            except Exception:
                raise
            matches = [s.strip().strip(".") for s in right.split(",")]
            res = by_index.get(res_idx)
            if res is None:
                raise

            if res.name == "CYS":
                if res in disulfide and "CYX" in matches:
                    choice = "CYX"
                elif "CYS" in matches:
                    choice = "CYS"
                elif "CYM" in matches:
                    choice = "CYM"
                else:
                    choice = matches[0]
            else:
                choice = matches[0]

            residue_templates[res] = choice

    raise RuntimeError("Could not resolve template ambiguity after 25 attempts.")


def run_simulation(
    forcefield,
    modeller,
    out_dir,
    temperature_k,
    allow_ignore_external_bonds=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system = build_system(
        forcefield,
        modeller,
        allow_ignore_external_bonds=allow_ignore_external_bonds,
    )
    system.addForce(MonteCarloBarostat(PRESSURE_ATM * atmosphere, temperature_k * kelvin))

    integrator = LangevinMiddleIntegrator(
        temperature_k * kelvin,
        FRICTION_PER_PS / picosecond,
        TIME_STEP_FS * femtoseconds,
    )

    platform = _get_platform()
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print("[MIN] minimizing energy")
    simulation.minimizeEnergy(maxIterations=MINIMIZE_MAX_ITERS)
    simulation.context.setVelocitiesToTemperature(temperature_k * kelvin)

    ca_idx = get_ca_atom_indices(modeller.topology)

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

    print("[EQUIL] starting")
    _run_stage(simulation, EQUIL_STEPS, REPORT_EVERY_STEPS, "EQUIL")

    # Record CA trajectory for production only so 1 ns @ 10 ps yields ~100 frames.
    simulation.reporters.append(HDF5Reporter(
        str(out_dir / "traj_ca.h5"),
        REPORT_EVERY_STEPS,
        atomSubset=ca_idx,
    ))
    print("[PROD] starting")
    _run_stage(simulation, PROD_STEPS, REPORT_EVERY_STEPS, "PROD")

    state = simulation.context.getState(getPositions=True)
    with open(out_dir / "final.pdb", "w") as f:
        PDBFile.writeFile(simulation.topology, state.getPositions(), f)

    return {"traj_ca_h5": "traj_ca.h5", "topology_ca_pdb": "topology_ca.pdb"}
