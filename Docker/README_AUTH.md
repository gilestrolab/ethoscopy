# JupyterHub Authentication Configuration Guide

This JupyterHub deployment supports multiple authentication methods, configurable via environment variables. Choose the authentication method that best fits your needs.

## Table of Contents

- [Quick Start](#quick-start)
- [Authentication Methods](#authentication-methods)
  - [Dummy (Shared Password)](#dummy-shared-password)
  - [GitHub OAuth](#github-oauth)
  - [Google OAuth](#google-oauth)
  - [GitLab OAuth](#gitlab-oauth)
  - [Keycloak / Generic OIDC](#keycloak--generic-oidc)
- [User Management](#user-management)
  - [Adding New Users (OAuth)](#adding-new-users-oauth)
  - [Adding New Users (Dummy Auth)](#adding-new-users-dummy-auth)
  - [Making Users Admins](#making-users-admins)
  - [Granting Cross-User File Access (Superusers)](#granting-cross-user-file-access-superusers)
- [Switching Authentication Methods](#switching-authentication-methods)
- [Dockerfile Optimization](#dockerfile-optimization)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

1. **Choose your authentication method** from the options below
2. **Copy the corresponding example file**:
   ```bash
   cp .env.dummy.example .env      # Shared password (default)
   cp .env.github.example .env     # GitHub OAuth
   cp .env.google.example .env     # Google OAuth
   cp .env.keycloak.example .env   # Keycloak/OIDC
   ```
3. **Edit `.env`** with your specific configuration
4. **Build and start the container**:
   ```bash
   docker compose build
   docker compose up -d
   ```

---

## Authentication Methods

### Dummy (Shared Password)

**Best for**: Lab environments, small teams, trusted networks

**Features**:
- Simple setup - no external services required
- All users share the same password
- Quick to deploy and test

**Configuration**:
```bash
cp .env.dummy.example .env
# Edit .env and set:
# - DUMMY_PASSWORD: The shared password
# - ALLOWED_USERS: Comma-separated list of usernames
# - ADMIN_USERS: Comma-separated list of admin usernames
```

**Security Note**: Not recommended for production or internet-facing deployments.

---

### GitHub OAuth

**Best for**: Open source projects, teams already using GitHub

**Setup Steps**:

1. **Create OAuth App in GitHub**:
   - Go to GitHub Settings → Developer Settings → OAuth Apps
   - Click "New OAuth App"
   - Fill in:
     - Application name: `JupyterHub - Ethoscope Lab`
     - Homepage URL: `http://your-server:8082`
     - Authorization callback URL: `http://your-server:8082/hub/oauth_callback`
   - Click "Register application"
   - Copy the **Client ID**
   - Generate and copy a **Client Secret**

2. **Configure environment**:
   ```bash
   cp .env.github.example .env
   # Edit .env and set:
   # - OAUTH_CLIENT_ID: From GitHub OAuth App
   # - OAUTH_CLIENT_SECRET: From GitHub OAuth App
   # - OAUTH_CALLBACK_URL: Must match GitHub settings
   # - ALLOWED_USERS: GitHub usernames (comma-separated)
   # - ADMIN_USERS: Admin GitHub usernames
   ```

3. **Optional - Restrict to Organization**:
   - Uncomment `GITHUB_ORG` in `.env`
   - Set to your organization name
   - Only organization members can log in

---

### Google OAuth

**Best for**: Organizations using Google Workspace, universities

**Setup Steps**:

1. **Create OAuth Credentials in Google Cloud Console**:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select existing
   - Enable **Google+ API**:
     - APIs & Services → Library → Search for "Google+" → Enable
   - Create Credentials:
     - APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID
     - Application type: **Web application**
     - Add Authorized redirect URI: `http://your-server:8082/hub/oauth_callback`
   - Copy the **Client ID** and **Client Secret**

2. **Configure environment**:
   ```bash
   cp .env.google.example .env
   # Edit .env and set:
   # - OAUTH_CLIENT_ID: From Google Cloud Console
   # - OAUTH_CLIENT_SECRET: From Google Cloud Console
   # - OAUTH_CALLBACK_URL: Must match Google settings
   # - ALLOWED_USERS: Email usernames without @domain.com
   # - ADMIN_USERS: Admin email usernames
   ```

3. **Optional - Restrict to Domain**:
   - Uncomment `GOOGLE_HOSTED_DOMAIN` in `.env`
   - Set to your domain (e.g., `university.edu`)
   - Only users from that domain can log in

---

### GitLab OAuth

**Best for**: Teams using GitLab, self-hosted GitLab instances

**Setup Steps**:

1. **Create OAuth Application in GitLab**:
   - Go to GitLab → User Settings → Applications
   - Fill in:
     - Name: `JupyterHub - Ethoscope Lab`
     - Redirect URI: `http://your-server:8082/hub/oauth_callback`
     - Scopes: Select `read_user`
   - Save application
   - Copy the **Application ID** and **Secret**

2. **Configure environment**:
   ```bash
   cp .env.gitlab.example .env
   # Edit .env and set:
   # - OAUTH_CLIENT_ID: GitLab Application ID
   # - OAUTH_CLIENT_SECRET: GitLab Secret
   # - OAUTH_CALLBACK_URL: Must match GitLab settings
   ```

3. **Optional - Self-hosted GitLab**:
   - Uncomment `GITLAB_URL` in `.env`
   - Set to your GitLab instance URL

---

### Keycloak / Generic OIDC

**Best for**: Organizations with existing SSO, enterprise environments

**Features**:
- Centralized user management
- Single Sign-On (SSO)
- Works with Keycloak or any OIDC-compliant provider

**Setup Steps for Keycloak**:

1. **Create Client in Keycloak**:
   - Log into Keycloak Admin Console
   - Select your realm (e.g., `gilestro-lab`)
   - Go to Clients → Create Client
   - Configure:
     - Client ID: `jupyterhub`
     - Client Protocol: `openid-connect`
     - Client Authentication: ON (confidential)
     - Valid Redirect URIs: `http://your-server:8082/hub/oauth_callback`
   - Save and go to **Credentials** tab
   - Copy the **Client Secret**

2. **Create Users in Keycloak**:
   - Go to Users → Add User
   - **CRITICAL**: Username MUST match existing home directory names exactly
   - Example: If home directory is `/home/ggilestro`, Keycloak username must be `ggilestro`
   - Create all required users from your `ALLOWED_USERS` list

3. **Configure environment**:
   ```bash
   cp .env.keycloak.example .env
   # Edit .env and replace:
   # - YOUR_KEYCLOAK_SERVER: e.g., http://keycloak.example.com
   # - YOUR_REALM: e.g., gilestro-lab
   # - OAUTH_CLIENT_SECRET: From Keycloak client credentials
   ```

4. **Keycloak OIDC Endpoints**:
   The URLs follow this pattern:
   ```
   http://YOUR_KEYCLOAK_SERVER/realms/YOUR_REALM/protocol/openid-connect/{auth,token,userinfo}
   ```

**Username Mapping**:
- Keycloak usernames MUST match home directory names exactly
- Use `preferred_username` claim (default)
- Example mapping:
  - Home dir: `/home/ggilestro` → Keycloak user: `ggilestro`
  - Home dir: `/home/labguest1` → Keycloak user: `labguest1`

---

## Switching Authentication Methods

To switch between authentication methods:

1. **Stop the container**:
   ```bash
   docker compose down
   ```

2. **Update `.env` file**:
   ```bash
   # Option 1: Copy a new example
   cp .env.github.example .env

   # Option 2: Edit existing .env
   nano .env
   # Change AUTHENTICATOR_CLASS and related settings
   ```

3. **Rebuild if needed** (only if changing providers for first time):
   ```bash
   docker compose build
   ```

4. **Start the container**:
   ```bash
   docker compose up -d
   ```

5. **Verify**:
   ```bash
   docker compose logs -f ethoscope-lab
   ```

---

## User Management

### Adding New Users (OAuth)

When using OAuth authentication (GitHub, Google, GitLab, Keycloak), adding a new user requires **two steps**:

#### Step 1: Add User to OAuth Provider

**For Keycloak (current setup at auth.gilest.ro)**:
1. Log in to Keycloak admin console: `https://auth.gilest.ro`
2. Select the `gilestro-lab` realm
3. Go to **Users** → **Add user**
4. Fill in:
   - **Username**: Must exactly match the desired home directory name (e.g., `jdoe`)
   - **Email**: User's email address
   - **First Name / Last Name**: User's full name
5. Click **Save**
6. Go to **Credentials** tab → Set password → Uncheck "Temporary"
7. Click **Save password**

**Important**: The Keycloak username MUST match the home directory name exactly (case-sensitive).

#### Step 2: Create Home Directory on Host

SSH to the server hosting JupyterHub and create the user's home directory:

```bash
# Create home directory (replace 'jdoe' with actual username)
sudo mkdir -p /mnt/homes/jdoe

# Set ownership to match container user (UID:GID from .env, default 1000:1000)
sudo chown 1000:1000 /mnt/homes/jdoe

# Set permissions
sudo chmod 755 /mnt/homes/jdoe

# Verify
ls -ld /mnt/homes/jdoe
# Should show: drwxr-xr-x 2 1000 1000 ... /mnt/homes/jdoe
```

**That's it!** The user can now log in through Keycloak and will automatically get access to JupyterHub.

#### Optional: Enforce Whitelist for OAuth

By default, **any user who authenticates via Keycloak can access JupyterHub** (allow_all=True). To restrict access to a whitelist:

1. **Edit `config/users.py`**:
   ```python
   ALLOWED_USERS = {
       'ggilestro',
       'jdoe',      # Add new user here
       'labguest1',
       # ... existing users
   }
   ```

2. **Edit `.env` and set**:
   ```bash
   ENFORCE_WHITELIST_FOR_OAUTH=true
   ```

3. **Restart container**:
   ```bash
   docker compose restart
   ```

### Adding New Users (Dummy Auth)

When using Dummy (shared password) authentication, adding a new user requires **three steps**:

#### Step 1: Add User to Whitelist

Edit `config/users.py` and add the username:

```python
ALLOWED_USERS = {
    'ggilestro',
    'jdoe',      # Add new user here
    'labguest1',
    # ... existing users
}
```

Or add to `.env` file:
```bash
ALLOWED_USERS=ggilestro,jdoe,labguest1,lguo
```

#### Step 2: Create Home Directory

Same as OAuth (Step 2 above):
```bash
sudo mkdir -p /mnt/homes/jdoe
sudo chown 1000:1000 /mnt/homes/jdoe
sudo chmod 755 /mnt/homes/jdoe
```

#### Step 3: Restart Container

```bash
docker compose restart
```

The new user can now log in with username `jdoe` and the shared password.

### Making Users Admins

Admin users can access the JupyterHub admin panel, start/stop other users' servers, and manage users.

#### Method 1: Edit config/users.py (Recommended)

```python
ADMIN_USERS = {
    'ggilestro',
    'jdoe',      # Add admin user here
}
```

#### Method 2: Edit .env file

```bash
ADMIN_USERS=ggilestro,jdoe
```

**Apply changes**:
```bash
docker compose restart
```

**Verify admin access**:
- Log in as the admin user
- Navigate to: `https://jupyter.lab.gilest.ro/hub/admin`
- You should see the admin panel with all users listed

### Granting Cross-User File Access (Superusers)

By default JupyterLab roots each user's file browser at their own home (`/home/<username>`). A **superuser** instead gets `/home` as their root, so they can browse, open, and run notebooks across every user's directory directly from their own JupyterLab session.

This works because all single-user servers in this deployment run under the same container UID (see `ConfigUserSpawner` in `config/jupyterhub_config.py`), so filesystem-level access to all homes already exists — `SUPERUSERS` simply surfaces it in the UI.

**Configuration**:

```bash
# .env
SUPERUSERS=ggilestro,jdoe
```

If `SUPERUSERS` is unset, it defaults to `ADMIN_USERS`.

**Apply changes**:
```bash
docker compose restart
```

**Verify**:
- Log in as a superuser
- The JupyterLab file browser should list every user's home folder
- Opening another user's `.ipynb` runs it in place; edits and kernel output write back to that user's home

**Security note**: Regular users are not isolated from each other at the filesystem level in this setup — anyone with a JupyterLab terminal can already `cd` into another user's home. `SUPERUSERS` only changes what is shown by default in the file browser. True per-user isolation would require a different spawner (e.g. `DockerSpawner` or `KubeSpawner` with distinct UIDs).

---

## Dockerfile Optimization

The Dockerfile has been optimized for fast rebuilds:

**Layer Order** (slow to fast):
1. ✅ System packages (apt-get) - rarely changes
2. ✅ R base installation - rarely changes
3. ✅ Python packages (including oauthenticator) - stable
4. ⚡ R packages - may change occasionally
5. ⚡ Ethoscope from source (git clone) - changes frequently

**Benefits**:
- Configuration changes don't rebuild all packages
- Git updates to Ethoscope only rebuild final layer
- Faster iteration during testing

**Build Tips**:
```bash
# Build with cache
docker compose build

# Force rebuild without cache (if needed)
docker compose build --no-cache

# Build specific service
docker compose build ethoscope-lab
```

---

## Troubleshooting

### Common Issues

#### 1. "Permission denied" or "User not found"
**Cause**: Username mismatch between authenticator and home directory

**Solution**:
- Ensure usernames in OAuth provider match home directory names exactly
- For Keycloak: Check `OAUTH_USERNAME_CLAIM` is set correctly
- Verify home directories exist: `ls /mnt/ethoscopy/homes/`

#### 2. "Redirect URI mismatch"
**Cause**: OAuth callback URL doesn't match provider settings

**Solution**:
- Check `OAUTH_CALLBACK_URL` in `.env`
- Verify it matches exactly in your OAuth provider (GitHub/Google/Keycloak)
- Include protocol (http/https), hostname, and port
- Example: `http://your-server:8082/hub/oauth_callback`

#### 3. "Client authentication failed"
**Cause**: Invalid client ID or secret

**Solution**:
- Verify `OAUTH_CLIENT_ID` and `OAUTH_CLIENT_SECRET` in `.env`
- Regenerate credentials in OAuth provider if needed
- Check for extra spaces or newlines in `.env` file

#### 4. Container fails to start
**Cause**: Missing or invalid environment variables

**Solution**:
```bash
# Check container logs
docker compose logs ethoscope-lab

# Verify .env file exists
cat .env

# Test configuration
docker compose config
```

#### 5. Users can't access their notebooks
**Cause**: ConfigUserSpawner not working properly

**Solution**:
- Check volume mounts: `docker compose config | grep volumes -A 5`
- Verify UID/GID matches host permissions
- Check home directory permissions: `ls -la /mnt/ethoscopy/homes/`

### Debug Mode

Enable verbose logging:

1. **Edit `config/jupyterhub_config.py`** and add:
   ```python
   c.JupyterHub.log_level = 'DEBUG'
   c.Authenticator.enable_auth_state = True
   ```

2. **Restart container**:
   ```bash
   docker compose restart ethoscope-lab
   ```

3. **View logs**:
   ```bash
   docker compose logs -f ethoscope-lab
   ```

### Testing Authentication

Test each component:

1. **Test OAuth endpoint connectivity**:
   ```bash
   # From inside container
   docker compose exec ethoscope-lab curl -v http://YOUR_KEYCLOAK_SERVER/realms/YOUR_REALM
   ```

2. **Verify environment variables**:
   ```bash
   docker compose exec ethoscope-lab env | grep OAUTH
   ```

3. **Check JupyterHub config**:
   ```bash
   docker compose exec ethoscope-lab cat /srv/jupyterhub/jupyterhub_config.py
   ```

---

## Security Best Practices

1. **Use HTTPS in production**:
   - Set up a reverse proxy (nginx, traefik)
   - Use Let's Encrypt for SSL certificates
   - Update all URLs to use `https://`

2. **Protect secrets**:
   - Never commit `.env` file to version control
   - Use `.env.*.example` files for templates only
   - Rotate OAuth secrets periodically

3. **Limit access**:
   - Use `ALLOWED_USERS` to whitelist specific users
   - Set up `ADMIN_USERS` carefully
   - For OAuth: use organization/domain restrictions

4. **Monitor access**:
   - Review logs regularly: `docker compose logs ethoscope-lab`
   - Check for failed login attempts
   - Monitor user activity

---

## Support

For issues specific to:
- **JupyterHub**: https://github.com/jupyterhub/jupyterhub/issues
- **OAuthenticator**: https://github.com/jupyterhub/oauthenticator/issues
- **Ethoscope Lab**: https://github.com/gilestrolab/ethoscope/issues

For Keycloak integration questions, see the integration plan in `KEYCLOAK_INTEGRATION_PLAN.md`.
