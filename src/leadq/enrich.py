"""Company homepage enrichment: fetch (fixture or live) + parse.

Offline fixtures (`fixtures/sites/<domain>.html`) are the default so demos and
evals are deterministic. `--live` fetches over HTTP with a timeout; transient
failures raise `TransientFetchError`, which the graph's RetryPolicy retries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import httpx
from bs4 import BeautifulSoup

from .models import SiteProfile

MAX_TEXT_CHARS = 6000
USER_AGENT = "leadq-enricher/0.1 (inbound lead qualification bot)"


class TransientFetchError(Exception):
    """Network-level failure worth retrying (timeout, connection reset, 5xx, 429)."""


class Fetcher(Protocol):
    source: str

    def fetch(self, domain: str, url: str) -> str | None:
        """Return HTML, None when there is no page (404 / no fixture), or raise TransientFetchError."""


class FixtureFetcher:
    source = "fixture"

    def __init__(self, sites_dir: Path) -> None:
        self.sites_dir = Path(sites_dir)

    def fetch(self, domain: str, url: str) -> str | None:
        if not domain:
            return None
        path = self.sites_dir / f"{domain}.html"
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")


class LiveFetcher:
    source = "live"

    def __init__(self, timeout: float = 8.0, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )

    def fetch(self, domain: str, url: str) -> str | None:
        target = url or (f"https://{domain}" if domain else "")
        if not target:
            return None
        try:
            resp = self.client.get(target)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientFetchError(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientFetchError(f"HTTP {resp.status_code} from {target}")
        if resp.status_code >= 400:
            return None
        if "html" not in resp.headers.get("content-type", "html"):
            return None
        return resp.text


# --------------------------------------------------------------------------
TECH_PATTERNS: dict[str, list[str]] = {
    "hubspot": ["js.hs-scripts.com", "hs-analytics", "hubspot"],
    "salesforce": ["salesforce", "pardot"],
    "intercom": ["widget.intercom.io", "intercom"],
    "zendesk": ["zdassets.com", "zendesk"],
    "segment": ["cdn.segment.com"],
    "stripe": ["js.stripe.com"],
    "shopify": ["cdn.shopify.com", "myshopify"],
    "wordpress": ["wp-content"],
    "webflow": ["webflow"],
    "slack": ["slack"],
    "zapier": ["zapier"],
    "pipedrive": ["pipedrive"],
    "google_analytics": ["googletagmanager.com", "gtag("],
}

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def detect_tech(html_lower: str) -> list[str]:
    return sorted(name for name, pats in TECH_PATTERNS.items() if any(p in html_lower for p in pats))


def parse_html(html: str, *, url: str, domain: str, source: str) -> SiteProfile:
    lower = html.lower()
    soup = BeautifulSoup(html, "html.parser")
    title = _clean(soup.title.get_text()) if soup.title else ""
    meta = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    meta_description = _clean(meta.get("content", "")) if meta else ""
    headings = [_clean(h.get_text(" ")) for h in soup.find_all(["h1", "h2", "h3"])]
    headings = [h for h in headings if h][:12]

    link_text = " ".join(
        f"{a.get_text(' ')} {a.get('href', '')}".lower() for a in soup.find_all("a")
    )
    for tag in soup(["script", "style", "noscript", "svg", "template"]):
        tag.decompose()
    text = _clean(soup.get_text(" "))[:MAX_TEXT_CHARS]
    text_lower = text.lower()
    return SiteProfile(
        url=url,
        domain=domain,
        found=True,
        source=source,  # type: ignore[arg-type]
        title=title,
        meta_description=meta_description,
        headings=headings,
        text=text,
        has_pricing="pricing" in link_text or "pricing" in text_lower or "/plans" in link_text,
        has_careers=any(k in link_text or k in text_lower for k in ("careers", "we're hiring", "jobs")),
        tech_hints=detect_tech(lower),
    )


def enrich(fetcher: Fetcher, domain: str, url: str) -> SiteProfile:
    html = fetcher.fetch(domain, url)  # may raise TransientFetchError -> retried by the graph
    if html is None:
        return SiteProfile(
            url=url,
            domain=domain,
            found=False,
            source="none",
            error="no homepage found" if domain else "no company domain (free email, no website)",
        )
    return parse_html(html, url=url, domain=domain, source=fetcher.source)
