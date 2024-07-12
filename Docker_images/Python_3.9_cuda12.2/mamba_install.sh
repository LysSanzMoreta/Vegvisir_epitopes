#!/bin/bash
micromamba env remove -n vegvisir
micromamba env create -n vegvisir -f env.yaml

