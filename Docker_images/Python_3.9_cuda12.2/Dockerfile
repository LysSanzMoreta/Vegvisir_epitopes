FROM mambaorg/micromamba:1.5-jammy-cuda-12.2.2 as micromamba39
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt
USER root

RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
#RUN micromamba install -y -n base -c conda-forge scvi-tools==1.0.4
#ENV PATH /opt/conda/bin:$PATH #retrieves pip to the environment
ENV PATH /usr/bin/apt:$PATH
RUN apt-get update
RUN apt-get install --yes wkhtmltopdf
RUN apt-get install graphviz
RUN apt-get install littler #installs Rscript
# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
#sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
#RUN apt-get update
#sudo apt install r-base
#RUN apt-get install libssl-dev
#RUN apt-get install libxml2
#RUN apt-get install libxml2-dev
##Install openssl with brew or not?
#RUN R -e "install.packages(c('ggseqlogo', 'ape', 'dplyr','tidyverse'), dependencies=TRUE,  repos='http://cran.rstudio.com/')" #needs to be sudo


#RUN module load apptainer
#RUN unset APPTAINER_BINDPATH
#RUN apptainer build --nv mymamba_gpu.sif Singularity.gpu
CMD ["/usr/bin/which","python3"] -> "/usr/bin/python3"
CMD ["/usr/bin/python3","/opt/.pycharm_helpers/packaging_tool.py","list"]
CMD ["/usr/bin/python3","/opt/.pycharm_helpers/remote_sync.py","--state-file","/tmp/e06641bc-ea0b-4a32-a780-2ed112e01659/.state.json","/tmp/bacecef2-e56a-4f46-9a40-fc180ff6cdd3"]

