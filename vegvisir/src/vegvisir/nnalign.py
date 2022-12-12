import subprocess,os,time
def run_nnalign(args,storage_folder,script_dir= "/home/projects/vaccine/people/lyssan/vegvisir_epitopes"):
    """:param str storage folder location of the data
       :param str script_dir Location of the data obtained from NNalign"""
    nnalign_location = "/home/projects/vaccine/people/morni/nnalign-2.1/nnalign"
    seq2logo_location = "/home/projects/vaccine/people/morni/seq2logo-2.1/seq2logo"

    train_data = "{}/{}/viral_nnalign_input_train.tsv".format(storage_folder,args.dataset_name)
    test_data = "{}/{}/viral_nnalign_input_eval.tsv".format(storage_folder,args.dataset_name)
    motif_len = 6
    working_dir = script_dir
    subprocess.Popen(args=[nnalign_location,"-f",train_data,"-name","viral1", "-testset",test_data, "-Logo", seq2logo_location,
                           "-lgt",motif_len,"-nh", 80, "-split",3,"-encoding",2,"-ishort","-fl",0,"-stop-iter",250],
                           cwd=working_dir,stdout=open(os.devnull, 'wb'),stderr=open(os.devnull, 'wb')) #stdin=PIPE, stderr=PIPE, stdout=PIPE


