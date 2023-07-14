from matplotlib.figure import Figure as matplotlib_figure
from plotly.graph_objs._figure import Figure as plotly_figure

def save_figure(fig, path, width = 1500, height = 650):
    """
    Figure could be of type matplotlib.figure.Figure for seaborn canvas or plotly.graph_objs._figure.Figure for plotly canvas
    Only plotly supports saving figures as html
    """

    assert(isinstance(path, str))
    
    if type(fig) == plotly_figure:

        if path.endswith('.html'):
           fig.write_html(path)
           print(f'Saved to {path}')

        else:
           fig.write_image(path, width=width, height=height)
           print(f'Saved to {path}')

    elif type(fig) == matplotlib_figure:
        if path.endswith('.html'):
            print ("Figure output not supported for %s" % path) 

        else:
            fig.savefig(path, dpi=200, bbox_inches='tight')
            print(f'Saved to {path}')

    else:
        print ("Figure type not supported or recognised.")