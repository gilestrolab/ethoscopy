# Dockerfile for jupyter-rethomics
ARG JUPYTER_HUB_TAG=5.4.0
FROM jupyterhub/jupyterhub:${JUPYTER_HUB_TAG}

# ============================================================================
# SYSTEM DEPENDENCIES AND TOOLS
# ============================================================================

# Install system prerequisites and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-dejavu \
        apt-utils \
        nano \
        jq \
        git \
        gfortran \
        file \
        gcc \
        cmake \
        libcurl4-openssl-dev \
        libfontconfig1-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libtiff-dev \
        libxml2-dev \
        libpam0g-dev \
        libssl-dev \
        libgit2-dev \
        libssh2-1-dev \
        libfreetype6-dev \
        libpng-dev \
        libjpeg-dev \
        libwebp-dev \
        libcap-dev \
        python3-dev \
        pkg-config \
        wget \
        lsb-release \
        software-properties-common \
        dirmngr \
        jupyter && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ============================================================================
# R INSTALLATION AND SETUP
# ============================================================================

# Add CRAN repository for latest R version
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc && \
    add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" -y

# Install R from CRAN repository
RUN apt-get update && \
    apt-get install -y r-base r-base-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install R packages and rethomics
COPY install_r_packages.r /root/install_r_packages.r
RUN ln -s /bin/tar /bin/gtar && \
    Rscript /root/install_r_packages.r

# ============================================================================
# JUPYTER INSTALLATION AND CONFIGURATION
# ============================================================================

# Install JupyterLab (version 3.6.4)
RUN pip3 install jupyterlab --ignore-installed

# To upgrade to version 4.0.2 (commented out)
# RUN pip3 install jupyterlab --upgrade

# Install Jupyter extensions (compatible with 3.x only)
RUN pip3 install jupyterlab-spreadsheet-editor

# ============================================================================
# PYTHON PACKAGES AND ETHOSCOPE SETUP
# ============================================================================

# Install Python data science packages
# Including oauthenticator for flexible authentication support
RUN pip3 install \
        oauthenticator \
        pandas \
        seaborn \
        bokeh \
        lifelines \
        opencv-python \
        mysql-connector \
        ethoscopy==2.1.0
       # pycatch22

# Pre-populate tutorial datasets inside the installed package so non-root
# JupyterHub users don't have to download them on first notebook run. The
# pickles (~36 MB total) live in the ethoscopy repo, not the PyPI wheel.
RUN python3 -c "from ethoscopy.misc.get_tutorials import download_tutorial_data, package_tutorial_data_dir; download_tutorial_data(dest_dir=package_tutorial_data_dir(), verbose=True)"

# Install Ethoscope from source
RUN git clone https://github.com/gilestrolab/ethoscope.git /opt/ethoscope && \
    cd /opt/ethoscope/ && git checkout dev && \
    cd /opt/ethoscope/src/ethoscope && pip install . 

# ============================================================================
# USER SETUP
# ============================================================================

# Add default user ethoscopelab with password ethoscopelab
RUN useradd -m ethoscopelab && \
    echo 'ethoscopelab:ethoscope' | chpasswd

# ============================================================================
# FINAL CONFIGURATION
# ============================================================================

WORKDIR /srv/jupyterhub/
# JupyterHub config will be mounted via docker-compose

EXPOSE 8000

LABEL maintainer="Giorgio Gilestro <giorgio@gilest.ro>" \
      org.jupyter.service="ethoscope-lab"

CMD ["jupyterhub"]
