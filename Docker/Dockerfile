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

# Install JupyterLab (latest 4.x; pip resolves the newest compatible release)
RUN pip3 install jupyterlab --ignore-installed

# To upgrade to version 4.0.2 (commented out)
# RUN pip3 install jupyterlab --upgrade

# Install Jupyter extensions (compatible with 3.x only)
RUN pip3 install jupyterlab-spreadsheet-editor

# AI assistant: Copilot-style inline completions + chat panel.
# Reason: lightweight (declares only jupyter-server as its Jupyter dep), so it
# does NOT pull in jupyter-collaboration / server-documents like jupyter-ai does
# — that RTC stack is what caused "File ID could not be retrieved" on file open.
# Pinned for reproducible builds. Supports OpenRouter via its OpenAI-compatible
# provider (base URL https://openrouter.ai/api/v1).
RUN pip3 install --no-cache-dir "notebook-intelligence==5.1.0"

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
        ipywidgets \
        ipympl \
        opencv-python \
        mysql-connector \
        scikit-learn \
        statsmodels \
        pingouin \
        openpyxl \
        pyarrow \
        tqdm \
        jupyterlab-git
       # pycatch22

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
# ETHOSCOPY (kept last on purpose)
# ============================================================================
# Reason: ethoscopy is the fastest-moving pin in this image and the one most
# often bumped for a bugfix. Everything above — the ~1 h R-package compile, the
# scientific pip stack, ethoscope-from-source — is stable, so installing
# ethoscopy in its own late layer means a version bump rebuilds only these two
# steps (seconds) instead of invalidating the whole pip block. Keep this the
# last software install; put nothing slow after it. Bumping the pin below is the
# only change needed to ship a new ethoscopy into the image.
RUN pip3 install ethoscopy==2.3.0

# Pre-populate tutorial datasets inside the installed package so non-root
# JupyterHub users don't have to download them on first notebook run. The
# pickles (~36 MB total) live in the ethoscopy repo, not the PyPI wheel.
#
# This is best-effort. urlretrieve() has no timeout, so a stalled connection to
# GitHub would hang the build forever (observed: a 36 MB / ~8 s download stuck
# >25 min); cap each connection with socket.setdefaulttimeout and retry, using
# overwrite so a socket killed mid-write cannot leave a partial file the next
# attempt skips as "already present". If it still cannot reach GitHub (e.g. a
# build host whose Docker bridge network is firewalled off from GitHub while the
# container's runtime network is not), warn and continue rather than failing the
# whole image: ethoscopy fetches the tutorial data on demand at first use, so a
# missing prefetch only costs a one-time download inside the user's session.
RUN for attempt in 1 2 3; do \
        python3 -c "import socket; socket.setdefaulttimeout(60); from ethoscopy.misc.get_tutorials import download_tutorial_data, package_tutorial_data_dir; download_tutorial_data(dest_dir=package_tutorial_data_dir(), overwrite=True, verbose=True)" && exit 0; \
        echo "tutorial download attempt $attempt failed; retrying in 5s..."; \
        sleep 5; \
    done; \
    echo "WARNING: could not pre-populate tutorial data at build time; it will download on first use at runtime." >&2

# ============================================================================
# FINAL CONFIGURATION
# ============================================================================

WORKDIR /srv/jupyterhub/
# JupyterHub config will be mounted via docker-compose

EXPOSE 8000

LABEL maintainer="Giorgio Gilestro <giorgio@gilest.ro>" \
      org.jupyter.service="ethoscope-lab"

CMD ["jupyterhub"]
