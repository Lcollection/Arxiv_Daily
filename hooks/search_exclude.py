from pathlib import PurePosixPath

from mkdocs.plugins import CombinedEvent, event_priority


_MISSING = object()
_NO_ATTR = object()
_PREVIOUS_SEARCH_META_ATTR = "_arxiv_daily_previous_search_meta"


def is_daily_paper_page(page) -> bool:
    src_path = PurePosixPath(page.file.src_uri)
    return src_path.parts[:1] == ("daily-papers",)


@event_priority(100)
def _mark_daily_papers_search_excluded(context, page, config, nav):
    if not is_daily_paper_page(page):
        return

    previous_search_meta = page.meta.get("search", _MISSING)
    setattr(page, _PREVIOUS_SEARCH_META_ATTR, previous_search_meta)

    search = dict(page.meta.get("search") or {})
    search["exclude"] = True
    page.meta["search"] = search


@event_priority(-100)
def _restore_daily_papers_search_meta(context, page, config, nav):
    if not is_daily_paper_page(page):
        return

    previous_search_meta = getattr(page, _PREVIOUS_SEARCH_META_ATTR, _NO_ATTR)
    if previous_search_meta is _NO_ATTR:
        return

    if previous_search_meta is _MISSING:
        page.meta.pop("search", None)
    else:
        page.meta["search"] = previous_search_meta

    delattr(page, _PREVIOUS_SEARCH_META_ATTR)


on_page_context = CombinedEvent(
    _mark_daily_papers_search_excluded,
    _restore_daily_papers_search_meta,
)
