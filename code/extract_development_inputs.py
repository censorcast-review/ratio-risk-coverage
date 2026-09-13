from pathlib import Path
from zipfile import ZipFile
import argparse
R=Path(__file__).resolve().parents[1];a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);p=a.parse_args();p.output.mkdir(parents=True,exist_ok=True)
with ZipFile(R/'forecast_archive/inputs/M5_DEVELOPMENT_INPUTS.zip') as z:
 for name in ['data/design_outcomes_v0_5.npz','data/calendar.csv','data/sell_prices.csv']:z.extract(name,p.output)
print('Extracted development-only inputs to',p.output)
