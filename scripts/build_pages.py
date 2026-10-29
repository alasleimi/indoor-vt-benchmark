"""Build a subpath-safe GitHub Pages site from the checked-in saved snapshot."""
from pathlib import Path
import shutil
import re


def build(output=Path('build/pages')):
    assets = Path('src/indoor_vt/execution/web')
    output.mkdir(parents=True, exist_ok=True)
    for path in assets.rglob('*'):
        if not path.is_file():
            continue
        target = output/path.relative_to(assets)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in ('.html', '.js', '.mjs', '.css') and 'vendor' not in path.parts:
            text = path.read_text(encoding='utf-8')
            # The live viewer is at /; Pages is mounted at /indoor-vt-benchmark/.
            for quote in ('"', "'", '`'):
                text = text.replace(quote+'/source?', quote+'./source.html?')
                text = text.replace(quote+'/construction?', quote+'./construction.html?')
            names = '|'.join(re.escape(p.name) for p in assets.iterdir() if p.is_file())
            if path.name != 'site-runtime.js':
                text = re.sub(r'''(["'`])/(?=(?:api/|vendor/|''' + names + r'''))''', r'\1./', text)
            text = text.replace('href="/"', 'href="./"').replace('href="/?', 'href="./?')
            if path.suffix == '.html':
                text = text.replace('<head>', '<head><script>window.VT_STATIC = true;</script>', 1)
            target.write_text(text, encoding='utf-8')
        else:
            shutil.copy2(path, target)
    shutil.copytree('published/redesign', output/'snapshot', dirs_exist_ok=True)
    (output/'.nojekyll').write_text('')


if __name__ == '__main__':
    build()
