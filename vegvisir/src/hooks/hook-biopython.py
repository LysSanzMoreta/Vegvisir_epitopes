from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import os
# Collect all submodules (optional)
# hiddenimports = (collect_submodules('Bio.Affy') +
#                  collect_submodules("Bio.Align") +
#                  collect_submodules("Bio.AlignIO") +
#                  collect_submodules("Bio.Alphabet") +
#                  collect_submodules("Bio.Application") +
#                  collect_submodules("Bio.Blast") +
#                  collect_submodules("Bio.CAPS") +
#                  collect_submodules("Bio.Cluster") +
#                  collect_submodules("Bio.codonalign") +
#                  collect_submodules("Bio.Compass") +
#                  collect_submodules("Bio.Data") +
#                  collect_submodules("Bio.Emboss") +
#                  collect_submodules("Bio.Entrez") +
#                  collect_submodules("Bio.ExPASy") +
#                  collect_submodules("Bio.GenBank") +
#                  collect_submodules("Bio.Geo") +
#                  collect_submodules("Bio.Graphics") +
#                  collect_submodules("Bio.HMM") +
#                  collect_submodules("Bio.KEGG") +
#                  collect_submodules("Bio.Medline") +
#                  collect_submodules("Bio.motifs") +
#                  collect_submodules("Bio.Nexus") +
#                  collect_submodules("Bio.NMR") +
#                  collect_submodules("Bio.Pathway") +
#                  collect_submodules("Bio.PDB") +
#                  collect_submodules("Bio.phenotype") +
#                  collect_submodules("Bio.Phylo") +
#                  collect_submodules("Bio.PopGen") +
#                  collect_submodules("Bio.Restriction") +
#                  collect_submodules("Bio.SCOP") +
#                  collect_submodules("Bio.SearchIO") +
#                  collect_submodules("Bio.SeqIO") +
#                  collect_submodules("Bio.Sequencing") +
#                  collect_submodules("Bio.SeqUtils") +
#                  collect_submodules("Bio.SVDSuperimposer") +
#                  collect_submodules("Bio.SwissProt") +
#                  collect_submodules("Bio.TogoWS") +
#                  collect_submodules("Bio.UniGene") +
#                  collect_submodules("Bio.UniProt") +
#                  collect_submodules("BioSQL")
#                  )
#


# Collect additional data files (if any)
#datas = collect_data_files(package="biopython",subdir='Bio.Align.substitution_matrices.data', includes=['*.txt', '*.csv','*'])
datas = collect_data_files('Bio.Align.substitution_matrices.data', includes=['*.txt', '*.csv','*'])


data_dir = os.path.join(os.path.dirname(__file__), '_internal/Bio/Align/substitution_matrices/data')
if os.path.exists(data_dir):
    datas += [(data_dir, 'Bio/Align/substitution_matrices/data')]