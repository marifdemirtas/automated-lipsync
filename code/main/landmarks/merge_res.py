import argparse
import pandas as pd

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('csv1', help='filename of the first CSV file')
parser.add_argument('csv2', help='filename of the second CSV file')
parser.add_argument('output', help='filename of the output CSV file')
args = parser.parse_args()

# read the two CSV files into separate data frames
df1 = pd.read_csv(args.csv1, names=['Filename', 'Distance', 'Crop'], delimiter=' ')
df2 = pd.read_csv(args.csv2, names=['Filename', 'Distance', 'Crop'], delimiter=' ')

df1['Filename'] = df1['Filename'].apply(lambda x: x.split('/')[-1])
df2['Filename'] = df2['Filename'].apply(lambda x: x.split('/')[-1])

# merge the two data frames on the 'Filename' column
merged = pd.merge(df1, df2, on='Filename')

merged = merged.rename(columns={
    'Distance_x': 'LQ_Distance',
    'Crop_x': 'LQ_Crop',
    'Distance_y': 'TR_Distance',
    'Crop_y': 'TR_Crop',
})
merged['Improvement'] = merged['LQ_Distance'] - merged['TR_Distance']

# save the merged data frame to a CSV file
merged.to_csv(args.output, index=False)
