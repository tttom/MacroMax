import sphinx.cmd.build
import sphinx.ext.apidoc
import shutil
import pathlib

from examples import log


log.info('Building the documentation...')
code_path = pathlib.Path(__file__).parent.parent.resolve()
docs_path = code_path / 'docs'
apidoc_path = docs_path / 'source/api'  # a temporary directory
html_output_path = docs_path / 'build/html'
log.info(f'Removing old documentation in {html_output_path}...')
shutil.rmtree(html_output_path, ignore_errors=True)
log.info('Building html...')
ret_value = sphinx.cmd.build.main(['-M', 'html', f"{docs_path / 'source'}", f"{docs_path / 'build'}"])
if ret_value != 0:
    log.error(f'sphinx-build returned {ret_value}.')
log.info(f'Removing temporary directory {apidoc_path}...')
shutil.rmtree(apidoc_path, ignore_errors=True)

build_path = docs_path / 'build/html/index.html'
log.info(f'Documentation landing page: {build_path}')

# Show the generated documentation
try:
    import webbrowser
    log.info(f'Opening documentation at {build_path}...')
    webbrowser.open(str(build_path))
except (ImportError, ModuleNotFoundError):
    log.info(f'Documentation ready, peruse at {build_path}')
