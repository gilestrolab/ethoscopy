# Ethoscopelab docker instance

Note: Most users will **not** need to recreate this image. These instructions are just provided as reference.
The ethoscopelab docker instance lives on dockerhub at the following address: [https://hub.docker.com/r/ggilestro/ethoscope-lab](https://hub.docker.com/r/ggilestro/ethoscope-lab) and this is what regular users should download and run. Follow instructions there and on the [ethoscopy manual](https://bookstack.lab.gilest.ro/books/ethoscopy/page/getting-started).


## Docker files that were used to create the ethoscope-lab docker instance

The files in this folder can be used to recreate the image as uploaded on dockerhub.

The command to use to recreate that image is `JUPYTER_HUB_TAG=5.3.0 ETHOSCOPE_LAB_TAG=1.0 docker compose build`. This creates the image with the specified tag. For Docker Hub deployment, push the image: `docker push ggilestro/ethoscope-lab:1.0`. To also create a latest tag: `docker tag ggilestro/ethoscope-lab:1.0 ggilestro/ethoscope-lab:latest && docker push ggilestro/ethoscope-lab:latest`. You can verify your local images with `docker images | grep ethoscope-lab`.

After creation, the image can be run using the enclosed `docker-compose.yml` file, replacing values as fit.

### The GPU does not speed up the build (it is a *runtime* resource)

A recurring confusion: **`docker compose build` cannot use the GPU, no matter what.**
The host does have an NVIDIA GPU (e.g. an RTX A4000) and the *running* container is
given access to it via the `deploy.resources.reservations.devices: capabilities: ["gpu"]`
block in `docker-compose.yml` — that is what GPU-aware analysis inside the notebook uses.
But building the image is an entirely separate phase that runs on CPU: BuildKit has no
GPU path, and every slow step here is CPU/IO-bound anyway.

What actually makes the build slow (~1–1.5 h from cold) is, in order:

1. **Compiling the 8 R packages** from source (`Rscript install_r_packages.r`) — by far
   the biggest cost, and pure CPU.
2. `pip install` of the JupyterLab / notebook-intelligence / scientific stack.
3. `apt-get` and the `git clone` of ethoscope.

None of these is GPU-accelerable. If a rebuild is taking the full hour+ it is almost
always because the **build cache was lost**, forcing the R compile to re-run. To avoid
re-paying it:

- **Do not kill a build mid-layer.** BuildKit only commits a layer's cache when that
  layer *finishes*; a build killed during the R step discards it and the next build
  recompiles from scratch.
- **Bumping `ethoscopy==X.Y.Z` only invalidates the pip layer and below.** The R compile
  sits earlier in the `Dockerfile`, so a version-pin bump should reuse the cached R layer
  and finish in minutes — *provided the cache still exists*. Old cache is eventually
  evicted (`docker buildx du` shows what is reclaimable), which is the usual reason an
  "only changed the pin" rebuild still takes an hour.
- **Prefer pulling over building.** Regular deployments should `docker pull
  ggilestro/ethoscope-lab:<tag>` rather than rebuild; only rebuild when the recipe itself
  changes, then push the result so nobody else has to.

Rule of thumb: reach for the GPU to make *analysis* faster, never to make the *image build*
faster — there is nothing to accelerate there.

## Authentication and User Management

This JupyterHub instance supports multiple authentication methods including:
- **Keycloak OAuth** (current production setup at auth.gilest.ro)
- GitHub OAuth
- Google OAuth
- GitLab OAuth
- Dummy authenticator (shared password)

### Adding New Users

**⚠️ Important**: Adding a new user requires steps on BOTH the authentication provider AND the JupyterHub host.

For detailed instructions on adding users, making users admins, and switching authentication methods, see:

**📖 [README_AUTH.md](./README_AUTH.md) - Complete Authentication and User Management Guide**

### Quick Reference

**For Keycloak (current setup)**:
1. Add user to Keycloak at `auth.gilest.ro` in the `gilestro-lab` realm
2. Create home directory on host: `sudo mkdir -p /mnt/homes/username && sudo chown 1000:1000 /mnt/homes/username`
3. (Optional) Add to `config/users.py` if using whitelist enforcement

**For Dummy Auth**:
1. Add username to `config/users.py` or `.env` file
2. Create home directory on host
3. Restart container

See [README_AUTH.md](./README_AUTH.md) for complete step-by-step instructions.

## Mounting Home Directories as Volumes

To persist user data and notebooks across container restarts, you should mount user home directories as Docker volumes. This is done by modifying the `docker-compose.yml` file.

### Benefits of mounting home directories:

1. **Data Persistence**: User notebooks, data files, and configurations survive container restarts and updates
2. **Backup and Recovery**: Easy to backup user data by copying the mounted directories
3. **Performance**: Direct access to host filesystem, avoiding container storage overhead
4. **Sharing**: Users can access their files from the host system if needed

### Example volume configuration:

Add volumes to your `docker-compose.yml`:

```yaml
services:
  jupyterhub:
    volumes:
      - ./user_data:/home  # Maps host ./user_data to container /home
      - ./jupyterhub_config.py:/srv/jupyterhub/jupyterhub_config.py
```

Or for individual user directories:

```yaml
volumes:
  - ./users/ggilestro:/home/ggilestro
  - ./users/amadabhushi:/home/amadabhushi
  - ./users/shared:/home/shared  # Shared directory for all users
```

This ensures all user work is preserved even when containers are recreated or updated.

## Database Preparation for Docker

When mounting ethoscope database files into Docker containers with read-only permissions (`:ro`), you may encounter "database disk image is malformed" errors. This occurs when SQLite databases are in WAL (Write-Ahead Logging) mode but mounted on read-only filesystems where WAL companion files (`.db-wal`, `.db-shm`) cannot be accessed.

### Understanding the Issue

**Why this happens:**
- Ethoscope databases may be created in **WAL mode** for better write performance
- WAL mode requires companion files (`.db-wal` and `.db-shm`) alongside the main `.db` file
- Docker read-only mounts (`:ro`) prevent SQLite from accessing these files properly
- This causes intermittent "database disk image is malformed" errors during data loading

**Symptoms:**
- Intermittent loading failures for specific ROIs
- Error message: "database disk image is malformed"
- Same ROI may succeed on some loads and fail on others
- More common with large databases (>1 GB)

### Solution: Convert Databases to DELETE Mode

Before mounting databases in Docker, convert them from WAL to DELETE mode:

#### Using the Python Script (Recommended)

```bash
# From the ethoscopy project root
python3 scripts/convert_wal_to_delete.py /mnt/ethoscope_data/results --dry-run

# Convert all databases with detailed output
python3 scripts/convert_wal_to_delete.py /mnt/ethoscope_data/results --verbose

# Force conversion even for recently modified files (use with caution)
python3 scripts/convert_wal_to_delete.py /mnt/ethoscope_data/results --force
```

#### Using the Bash Wrapper

```bash
# From the ethoscopy project root
./scripts/convert_databases.sh /mnt/ethoscope_data/results
```

#### Manual Conversion Using sqlite3

For individual databases:

```bash
sqlite3 /path/to/database.db "PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;"
```

For bulk conversion:

```bash
find /mnt/ethoscope_data/results -name "*.db" -type f -exec sqlite3 {} "PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;" \;
```

### Verification

After conversion, verify the database is in DELETE mode:

```bash
sqlite3 /path/to/database.db "PRAGMA journal_mode;"
# Should output: delete

sqlite3 /path/to/database.db "PRAGMA integrity_check;"
# Should output: ok
```

### Best Practices

1. **Run conversion BEFORE starting Docker containers**
   ```bash
   # From the ethoscopy project root, convert databases first
   ./scripts/convert_databases.sh /mnt/ethoscope_data/results

   # Then start containers
   cd Docker && docker compose up -d
   ```

2. **Do NOT convert databases while ethoscopes are actively writing**
   - The conversion script skips files modified in the last 24 hours by default
   - Use `--force` to override this safety check if needed

3. **Test on a backup first**
   ```bash
   # Create a test copy
   cp /path/to/database.db /tmp/test.db

   # Test conversion
   sqlite3 /tmp/test.db "PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;"

   # Verify it works
   sqlite3 /tmp/test.db "PRAGMA integrity_check;"
   ```

4. **Consider converting at the ethoscope source** (upstream fix)
   - Configure ethoscopes to use DELETE mode by default
   - Prevents the need for post-processing

### Troubleshooting

**Q: I still get "malformed" errors after conversion**

A: The ethoscopy library now includes improved connection handling that should work with both WAL and DELETE mode databases. If you still encounter issues:
1. Verify the database was actually converted: `sqlite3 database.db "PRAGMA journal_mode;"`
2. Check database integrity: `sqlite3 database.db "PRAGMA integrity_check;"`
3. Ensure you're using ethoscopy version 2.0.4 or later

**Q: Can I convert databases while Docker is running?**

A: Yes, but you should restart the container after conversion:
```bash
# From the ethoscopy project root
./scripts/convert_databases.sh /mnt/ethoscope_data/results
cd Docker && docker compose restart ethoscope-lab
```

**Q: Will this affect data quality or analysis?**

A: No. The conversion only changes how SQLite manages the database internally. All data remains identical and analysis results are unaffected.

**Q: How do I check if a database is in WAL mode?**

A:
```bash
sqlite3 database.db "PRAGMA journal_mode;"
```
Output will be either `wal` or `delete` (or `truncate`, `persist`, `memory`).

### Emergency Quick Fix

If you need to work immediately and can't run the full conversion:

```bash
# SSH to host machine (not inside container)
# Convert just the failing database
sqlite3 /path/to/failing/database.db "PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;"

# Restart container
docker compose restart ethoscope-lab
```
