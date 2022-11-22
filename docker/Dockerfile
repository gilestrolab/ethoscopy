#Dockerfile for jupyter-rethomics
FROM jupyterhub/jupyterhub

# pre-requisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-dejavu nano git gfortran file gcc cmake libcurl4-openssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# R
RUN apt-get update && \
    apt-get install -y libxml2-dev r-base && \
    rm -rf /var/lib/apt/lists/*

#install jupyter lab
RUN apt-get update && \
    apt-get install -y jupyter && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install jupyterlab --ignore-installed

#activate the multiuser jupyterlab in hub
COPY jupyterhub_config.py /srv/jupyterhub/jupyterhub_config.py

# Install Ethoscope and related packages
RUN pip3 install pandas seaborn bokeh lifelines
RUN pip3 install opencv-python mysql-connector
RUN cd /opt/ && \
    git clone https://github.com/gilestrolab/ethoscope.git && \
    cd ethoscope && git checkout dev && \
    cd src && python3 setup.py develop 

#add the default user ethoscolab with password ethoscolab
RUN useradd -m ethoscopelab
RUN echo 'ethoscopelab:ethoscopelab' | chpasswd

#install ethoscopy
#https://github.com/Lblackhurst29/ethoscopy
RUN pip3 install ethoscopy

#install R packages
COPY install_r_packages.r /root/install_r_packages.r
RUN ln -s /bin/tar /bin/gtar && \
    Rscript /root/install_r_packages.r

WORKDIR /srv/jupyterhub/

EXPOSE 8000

LABEL maintainer="Giorgio Gilestro <giorgio@gilest.ro>"
LABEL org.jupyter.service="ethoscope-lab"

CMD ["sh", "-c", "jupyterhub"]
