import numpy as np 
import re

import seaborn as sns
from plotly.express.colors import qualitative
from colour import Color

from ethoscopy.behavpy_core import behavpy_core

class behavpy_draw(behavpy_core):
    """
    Default drawing class containing some general methods that can be used by all children drawing classes
    """

    _hmm_colours = ['darkblue', 'dodgerblue', 'red', 'darkred']
    _hmm_labels = ['Deep sleep', 'Light sleep', 'Quiet awake', 'Active awake']

    @staticmethod
    def _check_rgb(lst):
        """ checks if the colour list is RGB plotly colours, if it is it changes it to its hex code """
        try:
            return [Color(rgb = tuple(np.array(eval(col[3:])) / 255)) for col in lst]
        except:
            return lst

    def _get_colours(self, plot_list):
        """ returns a colour palette from plotly for plotly """

        pl_len = len(plot_list)

        if self.canvas == 'plotly':
        
            if pl_len <= len(getattr(qualitative, self.attrs['sh_pal'])):
                return getattr(qualitative, self.attrs['sh_pal'])
            elif pl_len <= len(getattr(qualitative, self.attrs['lg_pal'])):
                return getattr(qualitative, self.attrs['lg_pal'])
            elif pl_len <= 48:
                return qualitative.Dark24 + qualitative.Light24
            else:
                raise IndexError('Too many sub groups to plot with the current colour palette (max is 48)')

        if self.canvas == 'seaborn':
            if pl_len <= len(list(sns.color_palette(self.attrs['sh_pal']))):
                return sns.color_palette(self.attrs['sh_pal'])
            else:
                return sns.color_palette('husl', len(plot_list))

    def _adjust_colours(self, colour_list):
        """ Takes a list of colours written names or hex codes.
        Returns two lists of hex colour codes. The first is a lighter version of the second which is the original.
        """
        def adjust_color_lighten(r,g,b, factor):
            return [round(255 - (255-r)*(1-factor)), round(255 - (255-g)*(1-factor)), round(255 - (255-b)*(1-factor))]

        colour_list = self._check_rgb(colour_list)

        start_colours = []
        end_colours = []
        for col in colour_list:
            c = Color(col)
            c_hex = c.hex
            end_colours.append(c_hex)
            r, g, b = c.rgb
            r, g, b = adjust_color_lighten(r*255, g*255, b*255, 0.75)
            start_hex = "#%02x%02x%02x" % (r,g,b)
            start_colours.append(start_hex)

        return start_colours, end_colours

    @staticmethod
    def _is_hex_color(s):
        """Returns True if s is a valid hex color. Otherwise False"""
        if re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', s):
            return True
        return False


    @staticmethod
    def _rgb_to_hex(rgb_string):
        """
        Takes a string defining an RGB color and converts it a string of equivalent hex
        Input should be a string containing at least 3 numbers separated by a comma.
        The following input will all work:
        rgb(123,122,100)
        123,122,100
        """

        # Only keep digits and commas
        filtered_string = ''.join(c for c in rgb_string if c.isdigit() or c == ',')

        # Split the filtered string by comma and convert each part to integer
        rgb_values = list(map(int, filtered_string.split(',')))

        # Map the values to integers
        r, g, b = map(int, rgb_values)

        # Convert RGB to hex
        hex_string = '#{:02x}{:02x}{:02x}'.format(r, g, b)

        return hex_string

    def save_figure(self, fig, path, width = None, height = None):
        assert(isinstance(path, str))

        if self.canvas == 'plotly':

            if path.endswith('.html'):
                fig.write_html(path)
            elif width is None and height is None:
                fig.write_image(path)
            else:
                fig.write_image(path, width=width, height=height)
            print(f'Saved to {path}')

        if self.canvas == 'seaborn':

            fig.savefig(path, bbox_inches='tight')
            print(f'Saved to {path}')
