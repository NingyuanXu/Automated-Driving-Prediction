import pandas as pd
from os import listdir
from os.path import join
from tqdm import tqdm


def main():
    print('consolidating train data')
    consol_fold('./train')  # your train data folder
    print('\n')
    
    print('consolidating test data')
    consol_fold('./val')  # your test data folder


# consolidate all the data into one csv
# the idx colum refers to which file it came from; you can c
# hange the name in for loop in the consol_data() function
def consol_fold(fold_path):
    X_path = fold_path + '/X/'
    Y_path = fold_path + '/y/'
    savePath = fold_path
    
    print('reading files....')
    df_X = consol_data(X_path)
    df_Y = consol_data(Y_path)
    
    df_X.to_csv(join(savePath, 'features.csv'), index=False)
    df_Y.to_csv(join(savePath, 'targets.csv'), index=False)


def consol_data(path):

    files = [join(path, f) for f in listdir(path)]
    
    files.reverse()
    
    ls_dfs = []
    # you can parallelelize this if too slow...
    for f in tqdm(files):
        sam_idx = [int(f[:-4].split('_')[-1])]
        tmp_df = pd.read_csv(f)
        tmp_df.insert(0, 'idx', sam_idx*len(tmp_df))
        ls_dfs.append(tmp_df)

    return pd.concat(ls_dfs)
 

if __name__ == '__main__':
    main()
    print('done')
