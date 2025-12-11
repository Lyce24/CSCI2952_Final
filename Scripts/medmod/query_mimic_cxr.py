import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import os
import shutil
import subprocess

medmod_fp = "/users/mspancho/Downloads/MedMod/mimic4extract/data/final/medmod_final.csv"

cmd = [
    "wget",
    "-r", "-N", "-c", "-np",
    "--user=mspancho",
    "--password=MIMICsp27!",
    "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/"
]

def main():
    print("Starting script")

    medmod_df = pd.read_csv(medmod_fp)
    # train_df, val_df, test_df = label_split(medmod_df)

    for row in medmod_df.itertuples():
        cmd_cxr = cmd.copy()
        cxr_path = str(row.cxr_path)
        cxr_path_correct = cxr_path[:4] + "p" + cxr_path[4:]
        cmd_cxr[7] = cmd_cxr[7] + str(cxr_path_correct)
        subprocess.run(cmd_cxr, check=True)

    print("Script complete")



if __name__ == "__main__":
    main()