import os
import dill
from tqdm import tqdm

data_dir = os.environ['PROJ_DATASETS']

outfile = open(os.path.join(data_dir, "dips_meta.dat"), 'w')
dips_raw_dir = "/media/data50/hayden/nih_project/external_tools/DIPS_plus/5134732/raw"

outfile.write('filename l_b l_len r_b r_len full\n')
wd = dips_raw_dir
directories = os.listdir(wd)
for directory in tqdm(directories, ascii=True):
    complxs = os.listdir(os.path.join(wd, directory))
    for complx_file in complxs:
        with open(os.path.join(wd, directory, complx_file), 'rb') as f:
            complx = dill.load(f)
            l_b = complx.sequences['l_b']
            l_len = str(len(l_b))
            r_b = complx.sequences['r_b']
            r_len = str(len(r_b))
            full = l_b + '|' + r_b
            outfile.write(' '.join([f"{directory}/{complx_file}", l_b, l_len, r_b, r_len, full]) + '\n')
outfile.close()
