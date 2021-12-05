import io
import os
import pandas as pd
from pandas.core.frame import DataFrame

RARE_THRESH = 0.01


        

def encode(snp):
    if snp == '0|0':
        return 1
    elif snp == '0|1':
        return 2
    elif snp == '1|0':
        return 3
    elif snp == '1|1':
        return 4
    else:
        return snp

def format_output(df:DataFrame) -> DataFrame:
    reduced_df = df.drop(columns=['CHROM', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'],).set_index('POS').T.sample(frac=1)
    return reduced_df.applymap(encode)

def selective(path:str, size:int, order:str, i_af:list):
    file_size = os.path.getsize(path)
    if i_af is None:
        i, i_af = 0, []
        with open(path, 'r') as f:
            #Find header row
            for l in f:
                print(l[:100])
                if l.startswith('#'):
                    continue
                info = l.split()[7].split(';')
                ac = float(info[0][3:])
                an = float(info[1][3:])
                i_af.append((i, ac/an))
                i += 1
                if i % 100 == 0:
                    print('{:.4f}%'.format(i*len(l)/file_size*100), end='\r')
        if len(i_af) < size:
            print('Only found {} SNPs'.format(len(i_af)))
            size = len(i_af)
    
    print('Sorting by AF...')
    if order == 'common':
        i_af.sort(key=lambda x:x[1], reverse=True)
    elif order == 'mid':
        i_af.sort(key=lambda x:abs(0.5-x[1]))
    elif order == 'rare':
        i_af.sort(key=lambda x:x[1])

    # Truncate to desired file size
    i_af_trunc = i_af[:size]

    print('Soring by index...')
    # Ascending sort by remaining index
    i_af_trunc.sort(key=lambda x:x[0])

    with open('data/{}_af_list_{}.txt'.format(order, size), 'w') as aff:
            aff.write(','.join([str(i_af_trunc[j][1]) for j in range(size)]))

    # Collect matching data
    i, j, lines = 0, 0, []
    with open(path, 'r') as f:
        #Find header row
        for l in f:
            if l.startswith('##'):
                continue
            if l.startswith('#CHROM'):
                lines.append(l[1:])
                continue
            if i_af_trunc[j][0] == i:
                lines.append(l)
                j += 1
                if j == size:
                    break
            i += 1
            if i % 100 == 0:
                print('{:.4f}%'.format(i*len(l)/file_size*100), end='\r')
    print(lines[0])
    df = format_output(
        pd.read_csv(
            io.StringIO(''.join(lines)),
            dtype={'CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str, 'QUAL': str, 'FILTER': str, 'INFO': str, 'FORMAT':str},
            sep='\t'))
    df.iloc[:int(0.8*len(df))].to_csv('data/{}_train_{}.tsv'.format(order, size), sep='\t')
    df.iloc[int(0.8*len(df)):].to_csv('data/{}_test_{}.tsv'.format(order, size), sep='\t')
    return i_af


def main():
    af = None
    for ws in [228, 2500]:
        for com_bool in ['rare', 'mid', 'common']:
            af = selective('data/ALL.chr22.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf', size=ws, order=com_bool, i_af=af)

if __name__ == '__main__':
    main()