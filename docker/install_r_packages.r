#!/bin/Rscript

install.packages(
c(
    'BiocManager',
    'caTools',
    'callr',
    'caret',
    'cowplot',
    'crayon',
    'curl',
    'DescTools',
    'devtools',
    'digest',
    'dplyr',
    'EnvStats',
    'forecast',
    'formatR',
    'ggplot2',
    'ggpubr',
    'ggtern',
    'ggthemes',
    'ggrepel',
    'ggseas',
    'gh',
    'git2r',
    'httr',
    'Hmisc',
    'IRkernel',
    'nycflights13',
    'openssl',
    'plotly',
    'r-plyr',
    'randomforest',
    'rcurl',
    'remotes',
    'reshape2',
    'rlang',
    'rmarkdown',
    'rsqlite',
    'Rtsne',
    'selectr',
    'shiny',
    'svglite',
    'stringi',
    'stringr',
    'survminer',
    'tictoc',
    'tidyr',
    'tidyverse',
    'usethis',
    'uuid',
    'wesanderson',
    'xgboost'
  ),
  repos='http://cran.uk.r-project.org'
)

# Install rethomics
install.packages(
c('behavr', 
  'ggetho', 
  'damr',
#  'scopr', 
  'sleepr', 
  'zeitgebr'
  ),
  repos= 'http://cran.uk.r-project.org'
)

#As of Jan 2024 scopr does not seem to be on CRAN for whatever reason so we need to install from github
library(devtools)
devtools::install_github("rethomics/scopr")

IRkernel::installspec(user = FALSE)
