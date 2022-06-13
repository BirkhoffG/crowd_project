from PIL import Image
import argparse
import pandas as pd
import numpy as np


def link2id(x):
    return x.split('/')[-1].split('"')[0]

def next_nan_label(df):
    return df.index[df['skin'].isnull()][0]

def main(args):
    img_pool_path = f"{args.img_pool}_skin.csv"
    img_dir = "img" if args.img_pool == 'fake' else 'original'
    img_pool_df = pd.read_csv(img_pool_path)

    while img_pool_df['skin'].isnull().values.any():
        idx = next_nan_label(img_pool_df)
        img_url = img_pool_df.loc[idx, 'INPUT:image']
        img_filename = link2id(img_url)
        img_path = f"{img_dir}/{img_filename}"
        img = Image.open(img_path)
        img.show()
        text = input(f"""
(id={idx}) Imge link: {img_url}
    [0] Lighter (DEFAULT)
    [1] Darker
Label the race:
        """)
        if not text:
            text = '0'
        if text not in ['0', '1']:
            raise ValueError('Input must be an integer from 0 to 1.')
        
        img_pool_df.loc[idx, 'skin'] = text
        img_pool_df.to_csv(img_pool_path, index=None)
        print("result store")

        img.close()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_pool', type=str, default='fake', choices=['org', 'fake'])
    args = parser.parse_args()
    main(args)

    