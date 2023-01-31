"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import subprocess,os,time
def run_nnalign(args,storage_folder,script_dir= "/home/projects/vaccine/people/lyssan/vegvisir_epitopes"):
    """:param str storage folder location of the data
       :param str script_dir Location of the data obtained from NNalign
       The format of the input datasets is specified at https://services.healthtech.dtu.dk/services/NNAlign-2.0/sampledata/HLA-DRB1.0101.s1000"""
    nnalign_location = "/home/projects/vaccine/people/morni/nnalign-2.1/nnalign"
    seq2logo_location = "/home/projects/vaccine/people/morni/seq2logo-2.1/seq2logo"

    train_data = "{}/{}/viral_nnalign_input_train.tsv".format(storage_folder,args.dataset_name)
    test_data = "{}/{}/viral_nnalign_input_valid.tsv".format(storage_folder,args.dataset_name)
    #motif_len = str(6)
    working_dir = script_dir
    for motif_len in range(6):
        subprocess.Popen(args=[nnalign_location,"-f",train_data,"-name","viral1", "-testset",test_data, "-Logo", seq2logo_location,
                               "-lgt",str(motif_len),"-nh", str(80), "-split",str(3),"-encoding",str(2),"-ishort","-fl",str(0),"-stop","-iter",str(250)],
                               cwd=working_dir)
                               #,stdout=open(os.devnull, 'wb'),stderr=open(os.devnull, 'wb')) #stdin=PIPE, stderr=PIPE, stdout=PIPE


