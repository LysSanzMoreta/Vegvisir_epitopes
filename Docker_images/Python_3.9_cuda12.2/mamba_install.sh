#!/bin/bash
micromamba env remove -n vegvisir
micromamba env create -n vegvisir
micromamba install -f env.yaml
