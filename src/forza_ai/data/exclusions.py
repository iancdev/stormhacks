"""Mandatory repository exclusions with a packaged fallback outside checkouts."""
from pathlib import Path


def default_policy_path():
    root = Path(__file__).resolve().parents[3]
    # In a checkout, deleting its policy must not silently select another policy.
    if (root / 'pyproject.toml').is_file():
        return root / 'config/exclude_sessions.txt'
    return Path(__file__).with_name('exclude_sessions.txt')


def exclusion_prefixes(additional=None):
    paths = [default_policy_path()]
    if additional is not None:
        paths.append(Path(additional))
    prefixes = set()
    for path in paths:
        for line in path.read_text().splitlines():
            prefix = line.split('#', 1)[0].strip()
            if prefix:
                prefixes.add(prefix)
    return sorted(prefixes)


def enforce_exclusions(*identities, additional=None):
    prefixes = exclusion_prefixes(additional)
    for identity in identities:
        if not isinstance(identity, str):
            continue
        source = identity.removeprefix('record.py:')
        if any(source.startswith(prefix) for prefix in prefixes):
            raise ValueError(f'session {identity!r} matches a mandatory exclusion prefix; '
                             'review config/exclude_sessions.txt before changing policy')
    return prefixes


def enforce_manifest(metadata):
    source = metadata.get('source_metadata', {})
    enforce_exclusions(metadata.get('session_id'), metadata.get('split_group'),
                       source.get('session') if isinstance(source, dict) else None)
