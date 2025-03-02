import pandas as pd
import statistics
import argparse

def calculate_avg_morph_length(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    segmentations = list(df[1].astype(str))
    morphs = []
    
    for segmentation in segmentations:
        ms = segmentation.replace("@@", "").split(" ")
        morphs.extend(ms)
    
    len_morphs = list(map(len, morphs))
    print(statistics.fmean(len_morphs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average morph length from a TSV file.")
    parser.add_argument("file_path", type=str, help="Path to the TSV file")
    args = parser.parse_args()
    
    calculate_avg_morph_length(args.file_path)




