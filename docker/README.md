# Ethoscopelab docker instance

Note: Most users will **not** need to recreate this image. These instructions are just provided as reference.
The ethoscopelab docker instance lives on dockerhub at the following address: [https://hub.docker.com/r/ggilestro/ethoscope-lab](https://hub.docker.com/r/ggilestro/ethoscope-lab) and this is what regular users should download and run. Follow instructions there and on the [ethoscopy manual](https://bookstack.lab.gilest.ro/books/ethoscopy/page/getting-started).


## Docker files that were used to create the ethoscope-lab docker instance

The files in this folder can be used to recreate the image as uploaded on dockerhub. 
The command to use to recreate that image is `sudo docker build -t ethoscope-lab .`
After creation, the image can be run using the following command, replacing values as fit. The environment variables are there to use the container together with nginxproxy/nginx-proxy and nginxproxy/acme-companion.

```
sudo docker run -d -p 8000:8000 \
      --name ethoscope-lab \
      --volume /mnt/data/results:/mnt/ethoscope_results:ro \
      --volume /mnt/homes/rstudio/:/home/ \
      --restart=unless-stopped \
      -e VIRTUAL_HOST="jupyter.lab.gilest.ro" \
      -e VIRTUAL_PORT="8000" \
      -e LETSENCRYPT_HOST="jupyter.lab.gilest.ro" \
      -e LETSENCRYPT_EMAIL="giorgio@gilest.ro" \
      ethoscope-lab
```

This will run an instance with local user control. To create new users from within the container use:

```
sudo docker exec -it ethoscope-lab /bin/bash
useradd new_user -m
passwd new_user
```

However, if you want to run a multiuser instance, it is reccomended to use persistant credentials information. After the first run and after having created new users from within the container, copy the relevant credential files to the host (e.g. the passwd folder) and re-run the container in the following way:

```
sudo docker run -d -p 8000:8000 \
      --name ethoscope-lab \
      --volume /mnt/data/results:/mnt/ethoscope_results:ro \
      --volume /mnt/data/ethoscope_metadata:/opt/ethoscope_metadata \
      --volume /mnt/homes:/home \
      --volume /mnt/cache:/home/cache \
      --restart=unless-stopped \
      -e VIRTUAL_HOST="jupyter.lab.gilest.ro" \
      -e VIRTUAL_PORT="8000" \
      -e LETSENCRYPT_HOST="jupyter.lab.gilest.ro" \
      -e LETSENCRYPT_EMAIL="giorgio@gilest.ro" \
      --volume /mnt/secrets/passwd:/etc/passwd:ro \
      --volume /mnt/secrets/group:/etc/group:ro \
      --volume /mnt/secrets/shadow:/etc/shadow:ro \
      --cpus=10 \
      ethoscope-lab

```

Note that in this latter case the credential files will be mounted as `ro` and can only be modified from the host machine, not from within the container.

The environment variables in the example below are used in conjunction with nginx-proxy and nginx-proxy-companion.

