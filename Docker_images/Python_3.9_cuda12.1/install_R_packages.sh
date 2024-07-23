#!/bin/sh
sudo apt-get install r-base-core
sudo apt install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev
sudo apt install libssl-dev libcurl4-openssl-dev libbz2-dev liblzma-dev libharfbuzz-dev and libfribidi-dev
sudo apt install -y libfontconfig1-dev libtiff5-dev
sudo Rscript -e "install.packages('devtools', repos='https://cran.rstudio.com/')"
sudo Rscript -e "devtools::install_github('omarwagih/ggseqlogo')"
sudo Rscript -e "install.packages('ape',dependencies=T)"
sudo Rscript -e "install.packages('dplyr',dependencies=T)"