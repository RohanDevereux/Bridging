from pdbfixer import PDBFixer
from openmm.app import Modeller, ForceField
from openmm.unit import nanometer, molar

from .config import FORCEFIELD_FILES, WATER_PADDING_NM, IONIC_STRENGTH_M

def load_and_fix(pdb_file):
    fixer = PDBFixer(filename=str(pdb_file))
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    return fixer

def select_chains(topology, positions, chain_ids):
    modeller = Modeller(topology, positions)
    keep = set(chain_ids)
    atoms_to_delete = [a for a in modeller.topology.atoms() if a.residue.chain.id not in keep]
    modeller.delete(atoms_to_delete)
    return modeller

def solvate(modeller, ph):
    forcefield = ForceField(*FORCEFIELD_FILES)
    modeller.topology.createDisulfideBonds(modeller.positions)
    modeller.addHydrogens(forcefield, pH=ph)
    modeller.addSolvent(
        forcefield,
        padding=WATER_PADDING_NM * nanometer,
        ionicStrength=IONIC_STRENGTH_M * molar,
    )
    return forcefield, modeller
