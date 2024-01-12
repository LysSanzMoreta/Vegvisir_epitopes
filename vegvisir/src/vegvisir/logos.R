
library(ggseqlogo)
library(ape)
library(dplyr)
library(tidyverse)
print("sucessfully running R script")
inputs <- commandArgs(trailingOnly = TRUE)
inputpath <- inputs[1]
outputpath <- inputs[2]
# Read peptide sequences from a file (peptides.txt)
peptides <- readLines(inputpath,warn = FALSE)
# Create custom coloring scheme

c_sty = make_col_scheme(chars=c('R', 'H', 'K', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W','#','-'),
                        groups=c('Basic', 'Basic', 'Basic', 'Acidic', 'Acidic', 'Polar', 'Polar', 'Neutral', 'Neutral', 'Basic', 'Polar', 'Hydrophobic', 'Hydrophobic', 'Hydrophobic', 'Hydrophobic', 'Hydrophobic', 'Hydrophobic', 'Hydrophobic', 'Hydrophobic', 'Hydrophobic','Gap','Gap'),
                        cols=c('blue', 'blue', 'blue', 'red', 'red', 'green', 'green', 'purple', 'purple', 'blue', 'green', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black','black','black'))
lg = ggseqlogo(peptides,col_scheme = c_sty)
ggsave(outputpath, lg, width = 8, height = 4, dpi = 300)



