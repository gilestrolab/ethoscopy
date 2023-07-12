#  Some spawners allow shell-style expansion here, allowing you to use
#  environment variables. Most, including the default, do not. Consult the
#  documentation for your spawner to verify!
#  Default: ['jupyterhub-singleuser']

#c.Spawner.cmd = ['jupyterhub-singleuser'] #Default would be single user
c.Spawner.cmd = ['jupyter-labhub']
