import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

medmod_fp = "/users/mspancho/Downloads/MedMod/mimic4extract/data/final/"
mimic_cxr_fp = "/users/mspancho/Downloads/physionet.org/files/mimic-cxr-jpg/2.1.0/"

def load_medmod(medmod_fp):
    stays_fp = os.path.join(medmod_fp, "all_stays.csv")
    labels_fp = os.path.join(medmod_fp, "phenotype_labels.csv")
    print("Reading MedMod CSVs")
    stays = pd.read_csv(stays_fp)
    stays['admittime'] = pd.to_datetime(stays['admittime'])
    stays['dischtime'] = pd.to_datetime(stays['dischtime'])
    stays['intime'] = pd.to_datetime(stays['intime'])
    stays['outtime'] = pd.to_datetime(stays['outtime'])
    labels = pd.read_csv(labels_fp)
    print("Concatenating MedMod CSVs")
    df = pd.concat([stays, labels], axis=1)
    print("Concatenated MedMod CSVs")

    df.drop(['last_careunit', 'los', 'race' , 'gender','age'], axis=1, inplace=True)
    print("Dropped unnecessary columns from MedMod dataframe")
    return df

def load_mimic_cxr_meta(mimic_cxr_fp):
    mimic_cxr_meta_fp = os.path.join(mimic_cxr_fp, "mimic-cxr-2.0.0-metadata.csv")
    print("Reading MIMIC-CXR metadata CSV")
    mimic_cxr_meta_df = pd.read_csv(mimic_cxr_meta_fp)
    print("Read MIMIC-CXR metadata CSV")
    mimic_cxr_meta_df['study_datetime'] = mimic_cxr_meta_df.apply(combine_date_time, axis=1)
    print("Combined date and time into study_datetime")
    return mimic_cxr_meta_df[['dicom_id', 'subject_id', 'study_id', 'StudyDate', 'StudyTime', 'study_datetime']]

def combine_date_time(row):
    date_str = str(row['StudyDate'])          # e.g. "21800506"
    time_str = str(row['StudyTime'])          # e.g. "213014.53100000002"

    # Split time into whole seconds and fractional seconds
    if '.' in time_str:
        whole, frac = time_str.split('.')
        frac = float("0." + frac)
    else:
        whole = time_str
        frac = 0.0

    # if len(whole) < 5: print(f"Unexpected StudyTime format: {time_str}")
    whole = whole.zfill(6)
    if len(whole) < 5: print(f"Unexpected StudyTime format: {time_str}")

    # Parse date + time (first 6 digits of time = HHMMSS)
    dt = pd.to_datetime(date_str + whole[:6], format="%Y%m%d%H%M%S")

    # Add fractional seconds
    dt = dt + pd.to_timedelta(frac, unit="s")

    return dt

def match_admission_to_cxr(row, mimic_cxr_meta_df):
    subject_id = row['subject_id']
    hadm_id = row['hadm_id']
    intime = row['intime']
    outtime = row['outtime']
    admittime = row['admittime']
    dischtime = row['dischtime']

    print(f"Matching subject_id: {subject_id}, hadm_id: {hadm_id}")

    cxr_for_subject = mimic_cxr_meta_df[mimic_cxr_meta_df['subject_id'] == subject_id]
    if cxr_for_subject.empty:
        print(f"No CXR found for subject_id: {subject_id}, hadm_id: {hadm_id}")
        return "No CXR found"

    if intime < admittime: firsttime = intime
    else: firsttime = admittime
    if outtime > dischtime: lasttime = outtime
    else: lasttime = dischtime
    # Filter studies that fall within the admission and discharge times
    matched_studies = cxr_for_subject[(cxr_for_subject['study_datetime'] >= firsttime) & (cxr_for_subject['study_datetime'] <= lasttime)]

    if matched_studies.empty:
        print(f"No CXR matched admission for subject_id: {subject_id}, hadm_id: {hadm_id}")
        return "No CXR matched admission"

    # Return the filename of the earliest matched study
    earliest_study = matched_studies.sort_values(by='study_datetime').iloc[0]
    # Drop to prevent reuse of the same image
    mimic_cxr_meta_df.drop(earliest_study.name, inplace=True)  
    # Format study to MIMIC-CXR-JPG file path
    earliest_study_path = f"p{str(subject_id)[:2]}/p{subject_id}/s{earliest_study['study_id']}/{earliest_study['dicom_id']}.jpg"

    return earliest_study_path

def link_medmod_to_cxr(medmod_df, mimic_cxr_meta_df):
    medmod_df['cxr_path'] = medmod_df.apply(lambda row: match_admission_to_cxr(row, mimic_cxr_meta_df), axis=1)

def main():
    print("Starting script")

    medmod_df = load_medmod(medmod_fp)
    mimic_cxr_meta_df = load_mimic_cxr_meta(mimic_cxr_fp)

    link_medmod_to_cxr(medmod_df, mimic_cxr_meta_df)
    
    filtered = medmod_df[medmod_df['cxr_path'].isin(["No CXR found", "No CXR matched admission"])]
    print(f"{len(filtered)} rows without matched CXRs")

    # Drop rows without matched CXRs
    medmod_df = medmod_df[~medmod_df['cxr_path'].isin(["No CXR found", "No CXR matched admission"])]

    # Save updated MedMod dataframe with CXR paths
    medmod_df.to_csv(os.path.join(medmod_fp, "medmod_final.csv"), index=False)

    print("Script complete")

if __name__ == "__main__":
    main()