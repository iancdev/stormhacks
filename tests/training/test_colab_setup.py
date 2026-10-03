import io
import json
from pathlib import Path
import stat
import zipfile

import pytest

from forza_ai.data.synthetic import generate

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / 'scripts/colab_archives.py'
namespace = {}
exec(compile(HELPERS.read_text(), str(HELPERS), 'exec'), namespace)
extract_zip = namespace['extract_zip']
extract_sessions = namespace['extract_sessions']
find_repository = namespace['find_repository']


def zip_tree(source, destination, prefix=''):
    with zipfile.ZipFile(destination, 'w') as archive:
        for path in source.rglob('*'):
            if path.is_file():
                archive.write(path, prefix + path.relative_to(source).as_posix())
    return destination


def test_notebook_helpers_match_and_cells_compile():
    notebook = json.loads((ROOT / 'notebooks/train_colab.ipynb').read_text())
    helper = next(c for c in notebook['cells'] if c['id'] == 'archive-helpers')
    assert ''.join(helper['source']) == HELPERS.read_text()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), cell['id'], 'exec')


@pytest.mark.parametrize('name', ['../escaped', '/absolute', 'C:/drive', 'a\\b', 'a/../../escaped'])
def test_archive_paths_rejected(tmp_path, name):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr(name, 'bad')
    with pytest.raises(ValueError, match='unsafe'):
        extract_zip(stream, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_symlink_and_case_collision_rejected(tmp_path):
    for symlink in (True, False):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            if symlink:
                member = zipfile.ZipInfo('link')
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(member, '../outside')
            else:
                archive.writestr('Frame.png', 'a')
                archive.writestr('frame.png', 'b')
        with pytest.raises(ValueError):
            extract_zip(stream, tmp_path / 'out')


def test_repository_zip_layout(tmp_path):
    source = tmp_path / 'source'
    (source / 'src/forza_ai/training').mkdir(parents=True)
    (source / 'src/forza_ai/training/cli.py').write_text('')
    (source / 'pyproject.toml').write_text('')
    archive = zip_tree(source, tmp_path / 'repo.zip', 'stormhacks-main/')
    root = extract_zip(archive, tmp_path / 'unpacked')
    assert find_repository(root) == root / 'stormhacks-main'
    with pytest.raises(ValueError, match='already exists'):
        extract_zip(archive, root)


def test_session_archives_transaction_and_layout(tmp_path):
    data = generate(tmp_path / 'fixtures', sessions=3, frames=4)
    one = zip_tree(data / 'synthetic-000', tmp_path / 'one.zip')
    two_source = tmp_path / 'two-source'
    two_source.mkdir()
    import shutil
    for name in ['synthetic-001', 'synthetic-002']:
        shutil.copytree(data / name, two_source / name)
    two = zip_tree(two_source, tmp_path / 'two.zip')
    summaries = extract_sessions([one, two], tmp_path / 'data')
    assert len(summaries) == 3
    assert sum(s['accepted'] for s in summaries) == 12
    with pytest.raises(ValueError, match='unique'):
        extract_sessions([one, one], tmp_path / 'duplicate')
    assert not (tmp_path / 'duplicate').exists()
    nested = zip_tree(data, tmp_path / 'nested.zip', 'extra-wrapper/')
    with pytest.raises(ValueError, match='layout'):
        extract_sessions([nested], tmp_path / 'nested')
    assert not (tmp_path / 'nested').exists()


def test_notebook_synthetic_cells_execute_locally(tmp_path):
    notebook = json.loads((ROOT / 'notebooks/train_colab.ipynb').read_text())
    cells = {c['id']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}
    scope = {}
    exec(cells['settings'].replace('/content/', str(tmp_path) + '/'), scope)
    for cell in ['archive-helpers', 'data', 'train', 'evaluate-export']:
        exec(cells[cell].replace('/content/', str(tmp_path) + '/'), scope)
    assert (scope['EXPORT'] / 'model.pt').is_file()
