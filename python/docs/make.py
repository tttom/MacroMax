import shutil
import warnings
from pathlib import Path

import sphinx.cmd.build


def build(destination_path: Path | str | None = None) -> Path:
    """
    Build documentation from the sources.

    Parameters:
        destination_path: The destination path for the documentation. Default: docs/build/

    Returns:
        The path to the generated documentation which contains a subdirectory 'html'.
    """
    source_path = Path(__file__).parent / 'src'
    apidoc_path = source_path / 'api'  # a temporary directory
    if destination_path is None:
        destination_path = source_path.parent / 'build'
    elif not isinstance(destination_path, Path):
        destination_path = Path(destination_path)
    html_root_path = destination_path / 'html'
    print(f'Building the documentation in {destination_path} from the sources in {source_path}...')

    print(f'Removing any old documentation in {html_root_path}...')
    shutil.rmtree(html_root_path, ignore_errors=True)
    html_index_path = html_root_path / 'index.html'
    print(f'Building html in {html_index_path.parent}...')
    ret_value = sphinx.cmd.build.main([str(_) for _ in ('-M', 'html', source_path, destination_path)])
    if ret_value != 0:
        warnings.warn(f'sphinx-build returned {ret_value}! Check if the documentation is ready at {html_index_path}.', stacklevel=2)
    else:
        print(f'Removing temporary directory {apidoc_path}...')
        shutil.rmtree(apidoc_path, ignore_errors=True)

        print(f'Documentation ready, peruse at {html_index_path}')

    return destination_path

def show(html_index_path: Path | str | None = None):
    """
    Show the generated documentation.

    Parameters:
        html_index_path: (optionally) The path to the file to open in a browser.
    """
    if html_index_path is None:
        html_index_path = Path(__file__).parent / 'build/html/index.html'
    elif not isinstance(html_index_path, Path):
        html_index_path = Path(html_index_path)
    try:
        import webbrowser
        print(f'Opening documentation at {html_index_path}...')
        if not webbrowser.open(str(html_index_path)):
            print(f'Error opening documentation at {html_index_path}.\nPlease try opening it manually.')
    except (ImportError, ModuleNotFoundError):
        print(f'No browser available to open documentation at {html_index_path}.')

if __name__ == '__main__':
    show(build() / 'html/index.html')
