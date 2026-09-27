"""crispz-studio - Tag autocomplete: the server-side sources.

Downloads the tag CSVs (the tag_autocomplete.sources config) into tags/ ONCE,
atomically and with console progress. Any .csv dropped into tags/ becomes a client-side
source. This module is imported ONLY when the feature is enabled (the zero-cost contract
when off); a network failure never blocks the boot (a warning + it carries on).

"""

import os

from cz_core import HERE, _log, download_with_progress

TAGS_DIR = os.path.join(HERE, "tags")


def _source_filename(url):
    """The local file name of a source URL (always .csv)."""
    from urllib.parse import urlparse
    name = os.path.basename(urlparse(str(url)).path.rstrip("/")) or "tags"
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return name


def ensure_tag_sources(sources):
    """Downloads every URL absent from tags/ (once). A failure = a warning, we carry on.
    Returns the number of files downloaded."""
    os.makedirs(TAGS_DIR, exist_ok=True)
    done = 0
    for url in (sources or []):
        dst = os.path.join(TAGS_DIR, _source_filename(url))
        if os.path.isfile(dst):
            continue
        try:
            _log(f"downloading {os.path.basename(dst)} (first launch only)...", mod="tagac")
            download_with_progress(url, dst)
            done += 1
        except Exception as e:
            _log(f"download failed for {url} ({e}); continuing without", mod="tagac")
    return done


def list_tag_files():
    """Every .csv of the tags/ folder (the client's sources)."""
    if not os.path.isdir(TAGS_DIR):
        return []
    return sorted(os.path.join(TAGS_DIR, f) for f in os.listdir(TAGS_DIR)
                  if f.lower().endswith(".csv"))
