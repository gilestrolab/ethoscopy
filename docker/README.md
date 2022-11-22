## Docker files to create the ethoscope-lab docker instance

Create an image with `sudo docker build -t ethoscope-lab.gilest.ro .`

Run the image using the following command, replacing values as fit.

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
      ethoscope-lab.gilest.ro
```

If you want to run a multiuser instance, it is reccomended to use you are using persistant credentials information.
After the first run and after having created new users from within the container, copy the relevant credential files to the host (e.g. the passwd folder)
and re-run the container in the following way:

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
      --volume /home/gg/mydockers/ethoscope-lab/passwd/passwd:/etc/passwd:ro \
      --volume /home/gg/mydockers/ethoscope-lab/passwd/group:/etc/group:ro \
      --volume /home/gg/mydockers/ethoscope-lab/passwd/shadow:/etc/shadow:ro \
      ethoscope-lab.gilest.ro
```

To create new users from within the container (reccomended) use:

```
sudo docker exec -it ethoscope-lab /bin/bash
useradd new_user -m
passwd new_user
```

