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
RUN apt-get install --yes graphviz
RUN apt-get install --yes littler #installs Rscript
RUN apt-get install --yes graphviz
RUN apt-get install --yes r-base-core
RUN apt install --yes libgtk-3-dev
RUN python -m pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04 -Iv wxPython==4.2.1 #necssary for installing Gooey
RUN apt install --yes build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev
RUN apt install --yes libssl-dev libcurl4-openssl-dev libbz2-dev liblzma-dev libharfbuzz-dev and libfribidi-dev
RUN apt install -y libfontconfig1-dev libtiff5-dev
RUN Rscript -e "install.packages('devtools', repos='https://cran.rstudio.com/')"
RUN Rscript -e "devtools::install_github('omarwagih/ggseqlogo')"
RUN Rscript -e "install.packages('ape',dependencies=T)"
RUN Rscript -e "install.packages('dplyr',dependencies=T)"



#Pycharm path oonfiguration
CMD ["/usr/bin/which","python3"] -> "/usr/bin/python3"
CMD ["/usr/bin/python3","/opt/.pycharm_helpers/packaging_tool.py","list"]
CMD ["/usr/bin/python3","/opt/.pycharm_helpers/remote_sync.py","--state-file","/tmp/e06641bc-ea0b-4a32-a780-2ed112e01659/.state.json","/tmp/bacecef2-e56a-4f46-9a40-fc180ff6cdd3"]

