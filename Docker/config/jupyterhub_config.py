#  Some spawners allow shell-style expansion here, allowing you to use
#  environment variables. Most, including the default, do not. Consult the
#  documentation for your spawner to verify!
#  Default: ['jupyterhub-singleuser']

import os
from jupyterhub.spawner import LocalProcessSpawner
from jupyterhub.auth import DummyAuthenticator

# Import OAuth authenticators (imported but may not be used depending on config)
try:
    from oauthenticator.github import GitHubOAuthenticator
    from oauthenticator.google import GoogleOAuthenticator
    from oauthenticator.gitlab import GitLabOAuthenticator
    from oauthenticator.generic import GenericOAuthenticator
except ImportError:
    # OAuth authenticators not available - only dummy auth will work
    pass

# Import user configuration from separate file
try:
    from users import ALLOWED_USERS, ADMIN_USERS
except ImportError:
    # Fallback if users.py doesn't exist
    ALLOWED_USERS = set()
    ADMIN_USERS = set()

#c.Spawner.cmd = ['jupyterhub-singleuser'] #Default would be single user
c.Spawner.cmd = ['jupyter-labhub', '--allow-root']

# ============================================================================
# FLEXIBLE AUTHENTICATION CONFIGURATION
# ============================================================================
# Configure authenticator via AUTHENTICATOR_CLASS environment variable
# Supported values: dummy (default), github, google, gitlab, generic (Keycloak/OIDC)

auth_class = os.getenv('AUTHENTICATOR_CLASS', 'dummy').lower()

if auth_class == 'github':
    # GitHub OAuth Authentication
    c.JupyterHub.authenticator_class = GitHubOAuthenticator
    c.GitHubOAuthenticator.client_id = os.getenv('OAUTH_CLIENT_ID')
    c.GitHubOAuthenticator.client_secret = os.getenv('OAUTH_CLIENT_SECRET')
    c.GitHubOAuthenticator.oauth_callback_url = os.getenv('OAUTH_CALLBACK_URL')
    # Allow all users who authenticate via OAuth (disable whitelist)
    c.GitHubOAuthenticator.allow_all = True
    # Optional: restrict to organization
    if os.getenv('GITHUB_ORG'):
        c.GitHubOAuthenticator.allowed_organizations = {os.getenv('GITHUB_ORG')}

elif auth_class == 'google':
    # Google OAuth Authentication
    c.JupyterHub.authenticator_class = GoogleOAuthenticator
    c.GoogleOAuthenticator.client_id = os.getenv('OAUTH_CLIENT_ID')
    c.GoogleOAuthenticator.client_secret = os.getenv('OAUTH_CLIENT_SECRET')
    c.GoogleOAuthenticator.oauth_callback_url = os.getenv('OAUTH_CALLBACK_URL')
    # Allow all users who authenticate via OAuth (disable whitelist)
    c.GoogleOAuthenticator.allow_all = True
    # Optional: restrict to hosted domain
    if os.getenv('GOOGLE_HOSTED_DOMAIN'):
        c.GoogleOAuthenticator.hosted_domain = os.getenv('GOOGLE_HOSTED_DOMAIN')

elif auth_class == 'gitlab':
    # GitLab OAuth Authentication
    c.JupyterHub.authenticator_class = GitLabOAuthenticator
    c.GitLabOAuthenticator.client_id = os.getenv('OAUTH_CLIENT_ID')
    c.GitLabOAuthenticator.client_secret = os.getenv('OAUTH_CLIENT_SECRET')
    c.GitLabOAuthenticator.oauth_callback_url = os.getenv('OAUTH_CALLBACK_URL')
    # Allow all users who authenticate via OAuth (disable whitelist)
    c.GitLabOAuthenticator.allow_all = True
    # Optional: custom GitLab URL
    if os.getenv('GITLAB_URL'):
        c.GitLabOAuthenticator.gitlab_url = os.getenv('GITLAB_URL')

elif auth_class == 'generic':
    # Generic OAuth (Keycloak, custom OIDC providers)
    c.JupyterHub.authenticator_class = GenericOAuthenticator
    c.GenericOAuthenticator.client_id = os.getenv('OAUTH_CLIENT_ID')
    c.GenericOAuthenticator.client_secret = os.getenv('OAUTH_CLIENT_SECRET')
    c.GenericOAuthenticator.oauth_callback_url = os.getenv('OAUTH_CALLBACK_URL')
    c.GenericOAuthenticator.authorize_url = os.getenv('OAUTH_AUTHORIZE_URL')
    c.GenericOAuthenticator.token_url = os.getenv('OAUTH_TOKEN_URL')
    c.GenericOAuthenticator.userdata_url = os.getenv('OAUTH_USERDATA_URL')
    c.GenericOAuthenticator.username_claim = os.getenv('OAUTH_USERNAME_CLAIM', 'preferred_username')
    # Allow all users who authenticate via OAuth (disable whitelist)
    c.GenericOAuthenticator.allow_all = True
    # Optional: scope customization
    if os.getenv('OAUTH_SCOPE'):
        c.GenericOAuthenticator.scope = os.getenv('OAUTH_SCOPE').split(',')

else:
    # Default: Simple authentication with shared password
    c.JupyterHub.authenticator_class = DummyAuthenticator
    c.DummyAuthenticator.password = os.getenv('DUMMY_PASSWORD', 'ethoscope')

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Allowed users - only apply whitelist for dummy authenticator by default
# OAuth authenticators use allow_all or provider-specific restrictions
# Set ENFORCE_WHITELIST_FOR_OAUTH=true in .env to enforce whitelist for OAuth too
enforce_whitelist_for_oauth = os.getenv('ENFORCE_WHITELIST_FOR_OAUTH', 'false').lower() == 'true'

if auth_class == 'dummy' or enforce_whitelist_for_oauth:
    allowed_users_env = os.getenv('ALLOWED_USERS')
    if allowed_users_env:
        # Environment override: parse comma-separated list
        c.Authenticator.allowed_users = set(allowed_users_env.split(','))
    else:
        # Use imported list from users.py
        c.Authenticator.allowed_users = ALLOWED_USERS

# Admin users - configurable via environment or users.py
admin_users_env = os.getenv('ADMIN_USERS')
if admin_users_env:
    # Environment override: parse comma-separated list
    c.Authenticator.admin_users = set(admin_users_env.split(','))
else:
    # Use imported list from users.py
    c.Authenticator.admin_users = ADMIN_USERS

# Superusers: users whose JupyterLab file browser is rooted at /home instead
# of their own home, so they can browse/open/run notebooks across every user.
# Reason: all single-user servers run as the same container UID (see
# ConfigUserSpawner below), so filesystem access already exists — this just
# exposes it in the UI. Defaults to admins; override with SUPERUSERS env var.
superusers_env = os.getenv('SUPERUSERS')
if superusers_env is not None:
    SUPERUSERS = {u.strip() for u in superusers_env.split(',') if u.strip()}
else:
    SUPERUSERS = set(c.Authenticator.admin_users)

# ============================================================================
# CUSTOM SPAWNER (Preserved - critical for current setup)
# ============================================================================
# Custom spawner that doesn't require system users

class ConfigUserSpawner(LocalProcessSpawner):
    def make_preexec_fn(self, name):
        """Don't try to switch users - run everything as current user"""
        return None

    def user_env(self, env):
        """Set user environment without system user lookup"""
        env = env.copy()
        home_dir = f'/home/{self.user.name}'

        # Ensure home directory exists with proper permissions
        os.makedirs(home_dir, mode=0o755, exist_ok=True)

        # Set environment variables
        env['USER'] = self.user.name
        env['HOME'] = home_dir
        env['SHELL'] = '/bin/bash'
        env['LOGNAME'] = self.user.name

        return env

c.JupyterHub.spawner_class = ConfigUserSpawner


def _set_notebook_dir(spawner):
    if spawner.user.name in SUPERUSERS:
        spawner.notebook_dir = '/home'
    else:
        spawner.notebook_dir = f'/home/{spawner.user.name}'


c.Spawner.pre_spawn_hook = _set_notebook_dir

# Timeouts
c.Spawner.http_timeout = 60
c.Spawner.start_timeout = 60
