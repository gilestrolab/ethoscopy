#!/bin/Rscript

install.packages(
c(
	 'EnvStats',
   'Hmisc',
	 'IRkernel',
	 'caTools',
	 'callr',
	 'caret',
   'cowplot',
	 'crayon',
	 'curl',
	 'devtools',
	 'digest',
	 'dplyr',
	 'forecast',
	 'formatR',
	 'ggplot2',
   'ggpubr',
   'ggtern',
	 'gh',
	 'git2r',
	 'httr',
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
	 'selectr',
	 'shiny',
	 'shiny',
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
  'scopr', 
  'sleepr', 
  'zeitgebr'
  ),
  repos= 'http://cran.uk.r-project.org'
)

# The following line is needed only for jupyter, not rstudio
IRkernel::installspec(user = FALSE)
