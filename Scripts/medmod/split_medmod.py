import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm
import os
import shutil
import subprocess

medmod_fp = "/users/mspancho/Downloads/MedMod/mimic4extract/data/final/medmod_final.csv"
mimic_dir = "/users/mspancho/scratch/physionet.org/files/mimic-cxr-jpg/2.1.0/files/"
save_dir = "/users/mspancho/scratch/medmod/"

# cmd = ["mv",f"{mimic_dir}",f"{save_dir}"]

def label_split(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
    # First split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Other upper respiratory disease'] # Most sparse label
    )

    # Then split train and val sets
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_df['Other upper respiratory disease']
    )

    return train_df, val_df, test_df

def main():
    print("Starting script")

    medmod_df = pd.read_csv(medmod_fp)
    train_df, val_df, test_df = label_split(medmod_df)

    split_dict = {
        "train": train_df,
        "val": val_df,
        "test": test_df}

    for split in split_dict:
        split_df = split_dict[split]

        split_dir = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        csv_path = os.path.join(split_dir, f"{split}.csv")

        # Move files belonging to this split
        new_paths = []
        missing_count = 0
        for row in tqdm(split_df.itertuples(), total=len(split_df), desc=f"Moving {split} files"):
            cxr_path = str(row.cxr_path)

            # Full absolute path to the source CXR image
            src = os.path.join(mimic_dir, cxr_path_correct)

            # Destination: /users/.../medmod/{split}/{filename}
            new_paths.append(os.path.basename(cxr_path_correct))
            dst = os.path.join(split_dir, os.path.basename(cxr_path_correct))

            # Skip if file already moved
            if os.path.exists(dst):
                continue

            if not os.path.exists(src):
                print(f"WARNING: Missing file {src}, skipping.")
                missing_count += 1
                continue

            # Move file
            shutil.move(src, dst)

        if missing_count > 0:
            print(f"WARNING: {missing_count} missing files in split '{split}'")

        split_df = split_df.copy()
        split_df['cxr_path'] = new_paths
        split_df.to_csv(csv_path, index=False)

        print(f"Script complete for {split} split")

        # split_df = split_dict[split]
        # csv_path = save_dir + split + "/" + split + ".csv"
        # split_df.to_csv(csv_path, index=False)

        # for row in split_df.itertuples():
        #     cmd_cxr = cmd.copy()
        #     cxr_path = str(row.cxr_path)
        #     cxr_path_correct = cxr_path[:4] + "p" + cxr_path[4:]
        #     cmd_cxr[1] = cmd_cxr[1] + str(cxr_path_correct)

        #     cxr_file_l = cxr_path_correct.split("/")
        #     cxr_file = cxr_file_l[-1]
        #     cmd_cxr[2] = cmd_cxr[2] + split + "/" + cxr_file

        #     if os.path.exists(cmd_cxr[2]):
        #         continue
        #     else:
        #         shutil.move(cmd_cxr[1], cmd_cxr[2])

    print("SCRIPT COMPLETE")

        

if __name__ == "__main__":
    main()