### Vegvisir (VAE): T-cell epitope classifier



If you find this library useful please cite: 

```

```



>INSTALLATION GUIDELINES:

Docker: Find the docker files under the  Docker_images folder

Isolated Python environment: `micromamba env create -n vegvisir -f env.yaml` where env.yaml is located in Docker_images/Python_3.*/env.yaml

R isolated environment: Install R 4.1.2 and packages following the script found at Docker_images/Python_3.*/install_R_packages.sh


> INPUT FILE: Using your own dataset

1. Create a TSV file with the following format:

Icore	target	partition	allele	Icore_non_anchor	target_corrected	confidence_score	Assay_number_of_subjects_responded	Assay_number_of_subjects_tested	allele_encoded	training	immunodominance_score	immunodominance_score_scaled	org_name
NANLWVTVY	0	0.0	HLA-C0702	NNLWVTV	0	0	0	1	8	True	0.0	0.0	0
FEEEKEWKTAV	0	0.0	HLA-B4001	FEEKEWKTA	0	0	0	1	6	True	0.0	0.0	0
RDINQTPFSF	1	0.0	HLA-B3701	RINQTPFS	1	0	0	1	2	True	0.0	0.0	0
IHLQYFECF	0	0.0	HLA-A2402	ILQYFEC	0	0	0	1	7	True	0.0	0.0	0
TCYPGHFADY	1	0.0	HLA-A2402	TCPGHFAD	1	0	0	1	7	True	0.0	0.0	0
AEWLDGDEEWL	1	0.0	HLA-B3701	AWLDGDEEW	1	0	0	1	2	True	0.0	0.0	0
FAYGKRHKDML	0	0.0	HLA-C0602	FYGKRHKDML	0	0	0	1	5	True	0.0	0.0	0

Compulsory column is <Icore> or <Icore_non_anchor>, if the other columns are not provided they will be assigned to zeroes, random values etc


2. Select args.dataset_name as `custom_dataset` and point to the file in args.train_path

> USING THE DATASETS and RESULTS from the publication

1. Select the following arguments and the run
   - args.dataset_name: `viral_dataset15`
   - args.config_dict: `BEST_hyperparameter_dict_blosum_vd15_z16.p` #best HPO configuration for the supervised dataset
2. If the necessary files are not available in the vegvisir/src/data folder, they will be downloaded there. If the download fails, these are the links:

   - ancho-info-content: https://drive.google.com/drive/folders/1kZScet33u6nC8eKURAAd1HYLUtbOyEP5?usp=sharing
   - common-files: https://drive.google.com/drive/folders/1kZScet33u6nC8eKURAAd1HYLUtbOyEP5?usp=sharing
   - viral_dataset15: "https://drive.google.com/drive/folders/1tPRGOJ0cQdLyW2GbdI2vnz1Sfr4RSKNf?usp=sharing



> USAGE

The main script that controls the model is Vegvisir_example.py and can be used in different modalities:

1. Training & Validation


2. Training & Testing


3. Sampling


4. Generating sequences (Working but under development). Requires larger training dataset and update on the NN architecture


5. Alteration of the sequence to modify its immnunogenicity (working but under development). Requires larger training dataset and update on the NN architecture

6. Benchmarking
   
    Please train the model first  using Hyperparameter optimization by setting the desired dataset, model and args.hpo == True. At the moment it tries to find if the GPU name starts with "accelerator", please change that if your device appears under a
7. different name when looking under *ray.available_resources()*

> FOLDERS MEANING
> 
> 
