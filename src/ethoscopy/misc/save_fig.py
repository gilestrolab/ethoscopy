def save_figure(fig, path, width = 1500, height = 650):
    assert(isinstance(path, str))
    if path.endswith('.html'):
        fig.write_html(path)
    else:
        fig.write_image(path, width=width, height=height)
    print(f'Saved to {path}')