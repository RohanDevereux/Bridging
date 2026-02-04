from pathlib import Path

from pandas import read_csv

package_dir = Path(__file__).resolve().parent.parent
data = read_csv(package_dir / "rawData" / "elife-07454-supp4.csv")
data["Baseline"] = data["ICs/NIS-based"]
data["Experimental"] = data["Binding_affinity"]
data["Error"] = data["Baseline"] - data["Experimental"]
cleaned_data = data[["PDB", "Baseline", "Experimental", "Error"]]
cleaned_data.to_csv(package_dir / "processedData" / "PRODIGY_Baseline.csv", index=False)
