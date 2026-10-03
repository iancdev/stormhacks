"""Stdlib-only Colab ZIP helpers, embedded in the notebook for bootstrap use.

Keep the notebook's archive-helpers cell in sync with this file. Only extract
archives whose code/data you trust; containment does not establish provenance.
"""
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


def extract_zip(archive, destination):
    """Extract to a new directory; reject traversal, links, devices and collisions."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'extraction destination already exists: {destination}')
    with zipfile.ZipFile(archive) as source:
        entries, seen = [], set()
        for member in source.infolist():
            name = member.filename
            relative = PurePosixPath(name)
            mode = member.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if (not name or '\\' in name or relative.is_absolute()
                    or '..' in relative.parts or ':' in name or not relative.parts):
                raise ValueError(f'unsafe ZIP path: {name!r}')
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f'ZIP links/special files are forbidden: {name!r}')
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f'ZIP path escapes destination: {name!r}')
            key = str(relative).casefold()
            if key in seen:
                raise ValueError(f'duplicate ZIP path: {name!r}')
            seen.add(key)
            entries.append((member, target))
        destination.mkdir(parents=True, exist_ok=False)
        try:
            for member, target in entries:
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as incoming, target.open('xb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
        except BaseException:
            shutil.rmtree(destination)
            raise
    return destination


def find_repository(extracted):
    """Accept a repository ZIP with either no wrapper or one GitHub wrapper."""
    extracted = Path(extracted)
    candidates = [extracted] + [p for p in extracted.iterdir() if p.is_dir()]
    candidates = [p for p in candidates if (p / 'pyproject.toml').is_file()
                  and (p / 'src/forza_ai/training/cli.py').is_file()]
    if len(candidates) != 1:
        raise ValueError('ZIP must contain exactly one repository with the training pipeline')
    return candidates[0]


def extract_sessions(archives, destination):
    """Transactionally unpack completed session ZIPs and validate every session.

    Each ZIP holds one session at its root, or session folders immediately below
    its root. Extraction happens on the destination filesystem, not Google Drive.
    """
    from forza_ai.data.sessions import load_sessions

    destination = Path(destination)
    archives = list(archives)
    if not archives or destination.exists():
        raise ValueError('provide at least one ZIP and a new destination directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='forza-unpack-', dir=destination.parent) as temporary:
        staging = Path(temporary)
        combined = staging / 'sessions'
        combined.mkdir()
        for index, archive in enumerate(archives):
            unpacked = extract_zip(archive, staging / f'archive-{index}')
            if (unpacked / 'metadata.json').is_file():
                candidates = [unpacked]
            else:
                candidates = list(unpacked.iterdir())
                if not candidates or any(not p.is_dir() for p in candidates):
                    raise ValueError(f'{archive}: expected session folders immediately below ZIP root')
            for candidate in candidates:
                required = ['metadata.json', 'frames.csv', 'wheel.csv', 'telemetry.csv']
                if not all((candidate / name).is_file() for name in required) or not (candidate / 'images').is_dir():
                    raise ValueError(f'{archive}: incomplete or nested session layout: {candidate.name}')
                # Use unique local names; session_id uniqueness is checked by the validator.
                target = combined / f'session-{len(list(combined.iterdir())):05d}'
                shutil.move(str(candidate), target)
        sessions = load_sessions(combined)
        if any(not session.samples for session in sessions):
            raise ValueError('archive contains a session with no eligible frames')
        summaries = [session.summary() for session in sessions]
        combined.rename(destination)
    return summaries
