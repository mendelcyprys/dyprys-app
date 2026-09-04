"""Command-line entry point for dyprys."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import textwrap
import time
from pathlib import Path

from dyprys import db
from dyprys import term
from dyprys import text as text_mod
from dyprys.chunker import OVERLAP_BYTES, TARGET_BYTES
from dyprys.check import survey
from dyprys.ingest import ingest_paths
from dyprys.routing import DEFAULT_BOOKS

# Why a result came back without its text. The three reasons need three
# different actions, and the one message they used to share named none of them.
_WHY_NO_TEXT = {
    text_mod.MISSING: "(the file is gone or its drive is not mounted; run `dyp check`)",
    text_mod.CHANGED: "(the file was edited and this passage is no longer in it;"
                      " re-run `dyp add` to re-index it)",
}


# Eighteen commands listed alphabetically tell a new reader nothing about which
# three they need today. argparse cannot group subcommands, so the flat listing
# is replaced by this, ordered by when in a library's life you reach for it.
COMMAND_GROUPS = """\
commands, in the order a library needs them

  first                 add      ingest text files
                        embed    turn them into vectors (resumable, slow)
                        route    profile each book so search can skip most of them

  every day             ask      find the passage that answers a question
                        asked    questions you have asked, and what came back
                        books    what is in the library
                        status   a summary, and what to do next

  while it is running   watch    follow an embedding run from another terminal
                        history  what has been done, and what it cost

  when something is off check    what has drifted and what work is outstanding
                        lexical  rebuild the keyword index
                        eval     measure retrieval against a question set

  looking after it      remove   forget books
                        compact  reclaim the space they left
                        relocate the text moved: rewrite a path prefix
                        models   embedding models: coverage, names, disk

  more than one         library  name and switch between indexes
                        backup   wrap an index and its text into one archive
                        restore  unpack one into an empty directory

`dyp COMMAND --help` for the flags of any of these.
"""


def _at_least_one(value: str) -> int:
    """A `-k` of 0 or less asks for no passages, which is a mistake, not a query.

    Without this the request fell through to an empty result set and was reported
    as "nothing embedded yet — run `dyp embed`", sending the user to re-embed a
    finished index over a bad argument.
    """
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    """Every command and flag, with no side effects.

    Separate from `main` so a test can read the registered commands back and
    check the grouped help still lists all of them.
    """
    parser = argparse.ArgumentParser(
        prog="dyp",
        description="Search a personal library of book-length texts.",
        epilog=COMMAND_GROUPS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", metavar="DIR",
        help="index location, by path",
    )
    parser.add_argument(
        "-L", "--library", metavar="NAME",
        help="index location, by registered name (see `dyp library`)",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    add = sub.add_parser("add", help="ingest text files into the index")
    add.add_argument("paths", nargs="+", type=Path, help="files or directories")
    add.add_argument(
        "--chapters",
        action="store_true",
        help="each given directory is one book, its files are chapters in "
        "filename order (the shape an EPUB extraction leaves)",
    )
    add.add_argument(
        "--ext", metavar="LIST", default=".txt",
        help="comma-separated file extensions to ingest (default: .txt). "
             "Text extraction from PDF or EPUB happens upstream.",
    )
    add.add_argument("--target", type=int, default=TARGET_BYTES, help="chunk size in bytes")
    add.add_argument("--overlap", type=int, default=OVERLAP_BYTES, help="chunk overlap in bytes")

    add.add_argument(
        "--deep",
        action="store_true",
        help="re-hash every file instead of trusting size and mtime",
    )

    embed = sub.add_parser("embed", help="embed outstanding chunks (resumable)")
    embed.add_argument(
        "--model", metavar="GGUF",
        help="embedding model file (default: $DYPRYS_MODEL)",
    )
    embed.add_argument(
        "--limit", type=int, metavar="N",
        help="stop after N chunks — for a bounded run",
    )
    embed.add_argument("--batch", type=int, default=8, help="chunks per model call")
    embed.add_argument(
        "--int8", action="store_true",
        help="store vectors as int8 — a quarter the disk, no measurable loss "
             "(only settable when a model is first registered)",
    )
    embed.add_argument(
        "--for", dest="duration", metavar="DURATION",
        help="stop after this long, e.g. 30m or 2h — finishes the batch in flight",
    )
    embed.add_argument(
        "--duty", type=int, metavar="PERCENT", default=100,
        help="share of the time to spend working, e.g. --duty 80 to leave the "
             "machine responsive — costs what it says, there is no free headroom")
    embed.add_argument(
        "--target", type=int, metavar="BYTES",
        help="embed the chunking of this size — only needed when a library has "
             "more than one (see `dyp books --chunkings`)")
    embed.add_argument(
        "-c", "--collection", metavar="PATTERN",
        help="embed only books whose title or path matches, e.g. -c neuroscience/",
    )

    ask = sub.add_parser("ask", help="search for the passage that answers a question")
    ask.add_argument("question")
    ask.add_argument("-k", type=_at_least_one, default=5, help="passages to return")
    ask.add_argument(
        "-c", "--collection", metavar="PATTERN",
        help="search only books whose title or path matches — a shelf "
             "(neuroscience/), one work (Kandel), or a set (\"*Imaging*\")",
    )
    ask.add_argument(
        "--mode", choices=("vector", "lexical", "hybrid"), default="hybrid",
        help="vector search, BM25, or both fused by rank (default)",
    )
    ask.add_argument(
        "--route", type=int, metavar="B", nargs="?", const=DEFAULT_BOOKS, default=None,
        help=f"route to the best B books first (default {DEFAULT_BOOKS} when given)",
    )
    ask.add_argument(
        "--dedupe", action="store_true",
        help="drop passages near-identical to a higher-ranked one — off by "
             "default, because in a library of several editions those are the "
             "comparison, not noise",
    )
    ask.add_argument(
        "--rerank", type=int, metavar="N", nargs="?", const=10, default=None,
        help="rescore the top N candidates with a cross-encoder (default 10)",
    )
    ask.add_argument("--reranker", metavar="GGUF", help="reranker model (default: $DYPRYS_RERANKER)")
    ask.add_argument(
        "--depth", type=int, metavar="N", default=0,
        help="retrieve N candidates before trimming to -k, without reranking — "
             "the arm that separates a deeper shortlist from a reordered one")
    ask.add_argument(
        "--expand", nargs="?", const=True, metavar="MODEL",
        help="rephrase the query with a model first, then search with every phrasing "
             "— off by default; see `dyp models` for what you can pass here")
    ask.add_argument("--expander", metavar="GGUF",
                     help="query expansion model (default: $DYPRYS_EXPANDER)")
    ask.add_argument("--quiet", "-q", action="store_true",
                     help="results only — no running commentary on stderr")
    ask.add_argument("--full", action="store_true",
                     help="print each passage whole, instead of its query-dense part")
    ask.add_argument("--json", action="store_true",
                     help="emit results as JSON for a program to parse, not the "
                          "human display")
    ask.add_argument(
        "--summarise", nargs="?", const=True, metavar="MODEL",
        help="draft an answer from the passages, keeping only quotations that "
             "verify against their source (default: $DYPRYS_SUMMARISER)")
    ask.add_argument("--model", metavar="GGUF", help="embedding model (default: $DYPRYS_MODEL)")

    ev = sub.add_parser("eval", help="measure answer recall and lexical safety")
    ev.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    ev.add_argument("-k", type=_at_least_one, default=5)
    ev.add_argument("--model", metavar="GGUF", help="embedding model (default: $DYPRYS_MODEL)")
    ev.add_argument("--lexical", type=int, default=20, help="exact-phrase probes")
    ev.add_argument("-c", "--collection", metavar="PATTERN", help="evaluate one shelf only")
    ev.add_argument(
        "--mode", choices=("vector", "lexical", "hybrid"), default="hybrid",
        help="vector search, BM25, or both fused by rank (default)",
    )
    ev.add_argument(
        "--route", type=int, metavar="B", nargs="?", const=DEFAULT_BOOKS, default=None,
        help=f"route to the best B books first (default {DEFAULT_BOOKS} when given)",
    )
    ev.add_argument(
        "--dedupe", action="store_true",
        help="drop passages near-identical to a higher-ranked one — off by "
             "default, because in a library of several editions those are the "
             "comparison, not noise",
    )
    ev.add_argument(
        "--rerank", type=int, metavar="N", nargs="?", const=10, default=None,
        help="rescore the top N candidates with a cross-encoder (default 10)",
    )
    ev.add_argument("--reranker", metavar="GGUF", help="reranker model (default: $DYPRYS_RERANKER)")
    ev.add_argument(
        "--depth", type=int, metavar="N", default=0,
        help="retrieve N candidates before trimming to -k, without reranking — "
             "the arm that separates a deeper shortlist from a reordered one")
    ev.add_argument("--expand", nargs="?", const=True, metavar="MODEL",
                    help="rephrase the query with a model before searching")
    ev.add_argument("--expander", metavar="GGUF",
                    help="query expansion model (default: $DYPRYS_EXPANDER)")
    ev.add_argument("--compare", action="store_true", help="score all three modes")

    bk = sub.add_parser("books", help="list books, or show one in detail")
    bk.add_argument("pattern", nargs="?", help="show matching books in full")
    bk.add_argument("--json", action="store_true", help="emit as JSON for a program to parse")

    md = sub.add_parser("models", help="embedding models: coverage and disk")
    md.add_argument("--json", action="store_true", help="emit as JSON for a program to parse")
    for _role, _env, _flag, _key, _what in (
        ("expansion", "DYPRYS_EXPANDER", "", "expander", ""),
        ("summarising", "DYPRYS_SUMMARISER", "", "summariser", ""),
        ("reranking", "DYPRYS_RERANKER", "", "reranker", ""),
    ):
        md.add_argument(f"--{_key}", metavar="MODEL",
                        help=f"remember a default {_key} for this library "
                             f"('none' to forget)")
    md.add_argument("--drop", metavar="NAME", help="remove a model and its vectors")
    md.add_argument("--yes", action="store_true", help="confirm a --drop")
    md.add_argument("--quantise", metavar="NAME", help="convert a model's vectors to int8")
    md.add_argument("--truncate", metavar="N", type=int,
                    help="keep only the first N dimensions — Matryoshka models only")
    md.add_argument("--name", nargs=2, metavar=("MODEL", "ALIAS"),
                    help="give a model a short name to type on the command line")
    md.add_argument("--needed", action="store_true",
                    help="what weights this index requires, and how to verify them")
    md.add_argument("--verify", metavar="GGUF", type=Path,
                    help="check a candidate file against what this index needs")
    md.add_argument("--source", nargs=2, metavar=("NAME", "URI"),
                    help="record where a model's weights can be obtained")
    sub.add_parser("lexical", help="build the BM25 index for an existing library")
    bu = sub.add_parser("backup", help="wrap the index and its text into one archive")
    bu.add_argument("archive", type=Path, help="where to write it, e.g. library.tar.gz")
    bu.add_argument("--no-sources", action="store_true",
                    help="index only — smaller, but passages cannot be shown elsewhere")

    rs = sub.add_parser("restore", help="unpack a backup into an empty index")
    rs.add_argument("archive", type=Path)
    rs.add_argument("--into", type=Path, required=True, help="empty directory for the index")
    rs.add_argument("--sources", type=Path, help="where to put the source text")
    rs.add_argument("--as", dest="register", metavar="NAME",
                    help="register the restored index under this name")

    mv = sub.add_parser("relocate", help="the library moved: rewrite a path prefix")
    mv.add_argument("old", help="the prefix as recorded")
    mv.add_argument("new", help="where it lives now")

    rm = sub.add_parser("remove", help="forget books; their chunks become dead space")
    rm.add_argument("pattern", help="books whose title or path matches")
    rm.add_argument("--yes", action="store_true", help="confirm")

    cp = sub.add_parser("compact", help="reclaim dead chunk ids across every store")
    cp.add_argument("--yes", action="store_true", help="confirm")

    rt = sub.add_parser("route", help="profile each book for two-stage search")
    rt.add_argument("--centroids", type=int, default=16, help="directions per book")
    rt.add_argument("--model", metavar="GGUF", help="embedding model (default: $DYPRYS_MODEL)")

    lib = sub.add_parser("library", help="name the libraries this installation knows")
    libsub = lib.add_subparsers(dest="action")
    la = libsub.add_parser("add", help="register a directory under a name")
    la.add_argument("name"); la.add_argument("path", type=Path)
    ll = libsub.add_parser("list", help="every library, and which is the default")
    ll.add_argument("--json", action="store_true", help="emit as JSON for a program to parse")
    lr = libsub.add_parser("remove", help="forget a name; the files stay")
    lr.add_argument("name")
    lr.add_argument("--delete", action="store_true",
                    help="also delete the index directory — not the source text")
    lr.add_argument("--yes", action="store_true", help="confirm a --delete")
    lu = libsub.add_parser("use", help="make one the default")
    lu.add_argument("name")

    st = sub.add_parser("status", help="what is in the index")
    st.add_argument("--json", action="store_true", help="emit as JSON for a program to parse")
    ak = sub.add_parser("asked", help="questions you have asked, and what came back")
    ak.add_argument("which", nargs="?", type=int,
                    help="show one in full, by the number `dyp asked` gives it")
    ak.add_argument("-n", "--limit", type=int, default=15, metavar="N",
                    help="how many to list (default 15)")
    ak.add_argument("--find", metavar="TEXT", help="only questions containing TEXT")
    ak.add_argument("--forget", metavar="WHAT",
                    help="'all', a number, or a date like 2026-08-01 to drop "
                         "everything older")
    ak.add_argument("--yes", action="store_true", help="confirm a --forget")
    ak.add_argument("--json", action="store_true", help="emit as JSON for a program to parse")

    hi = sub.add_parser("history",
                        help="what has been done to this index, and what embedding cost")
    hi.add_argument("-n", "--limit", type=int, default=15, metavar="N",
                    help="how many of the most recent to show (default 15)")
    hi.add_argument("--json", action="store_true", help="emit as JSON for a program to parse")
    wa = sub.add_parser("watch", help="follow an embedding run already in progress")
    wa.add_argument("--every", type=float, default=2.0, metavar="SECONDS",
                    help="how often to refresh (default 2)")

    check = sub.add_parser("check", help="what has drifted and what work is outstanding")
    check.add_argument("--json", action="store_true", help="emit as JSON for a program to parse")
    check.add_argument(
        "--deep",
        action="store_true",
        help="re-hash every file instead of trusting size and mtime",
    )

    # The grouped epilog above is the listing; argparse's own flat one would
    # print all eighteen a second time. The per-command `help=` strings are kept
    # rather than SUPPRESSed, so shell completion and any other tool reading the
    # parser still sees them.
    listing = getattr(sub, "_choices_actions", None)
    if listing is not None:
        listing.clear()
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "library":
        return _library(args, parser)

    where = _where(args)
    # Only these can bring an index into being. Everything else asking for one
    # that is not there means a wrong path, a stale library entry, or a drive
    # that is not mounted -- and creating an empty index at that spot hides all
    # three behind "nothing embedded yet".
    if args.command not in ("add", "restore") and not db.index_exists(where):
        named = f" (library {args.library!r})" if getattr(args, "library", None) else ""
        print(f"no index at {where}{named}", file=sys.stderr)
        print("`dyp add` creates one; `dyp library list` shows what is registered.",
              file=sys.stderr)
        return 2

    try:
        conn = db.connect(where)
    except ValueError as wrong:
        # A refusal to open is a decision, not a crash: say what is wrong and
        # what to do, rather than handing the user a traceback.
        print(f"cannot open the index: {wrong}", file=sys.stderr)
        return 2
    try:
        if args.command == "add":
            return _add(conn, args)
        if args.command == "history":
            return _history(conn, args.limit, args.json)
        if args.command == "asked":
            return _asked(conn, args)
        if args.command == "watch":
            return _watch(conn, _where(args), args.every)
        if args.command == "status":
            return _status(conn, args.json)
        if args.command == "check":
            return _check(conn, args.deep, args.json)
        if args.command == "embed":
            return _embed(conn, _where(args), args)
        if args.command == "ask":
            return _ask(conn, _where(args), args)
        if args.command == "eval":
            return _eval(conn, _where(args), args)
        if args.command == "books":
            return _books(conn, args.pattern, args.json)
        if args.command == "models":
            return _models(conn, _where(args), args)
        if args.command == "lexical":
            return _lexical(conn)
        if args.command == "route":
            return _route(conn, _where(args), args)
        if args.command == "backup":
            return _backup(conn, _where(args), args)
        if args.command == "restore":
            return _restore(args)
        if args.command == "relocate":
            return _relocate(conn, args)
        if args.command == "remove":
            return _remove(conn, args)
        if args.command == "compact":
            return _compact(conn, _where(args), args)
    finally:
        conn.close()
    return 0


def _backup(conn, directory, args) -> int:
    """Everything needed to search this library on another machine."""
    from dyprys.backup import write

    started = time.time()

    def show(done, total):
        print(f"\r  {done:,}/{total:,} source files packed ", end="", file=sys.stderr, flush=True)

    report = write(conn, directory, args.archive,
                   include_sources=not args.no_sources, progress=show)
    print(file=sys.stderr)
    print(f"wrote {report.path}  {_size(report.bytes_written)}")
    print(f"  {report.books:,} books, {report.chunks:,} chunks, {report.models} model(s)"
          f"{'' if args.no_sources else f', {report.sources:,} source files'}")
    print(f"  in {time.time() - started:.1f}s")
    if args.no_sources:
        print("\nsources were not included: restored elsewhere this can rank passages "
              "but not display them.")
    return 0


def _restore(args) -> int:
    """Unpack into an empty directory and point it at the text."""
    from dyprys.backup import read_manifest, restore

    try:
        manifest = read_manifest(args.archive)
    except (OSError, ValueError) as bad:
        print(f"cannot read {args.archive}: {bad}", file=sys.stderr)
        return 2

    if manifest.get("includes_sources") and not args.sources:
        print("this backup carries its source text; pass --sources DIR to say where "
              "it should go", file=sys.stderr)
        return 2

    def show(done, total):
        print(f"\r  {done:,}/{total:,} source files unpacked ", end="", file=sys.stderr, flush=True)

    try:
        report = restore(args.archive, args.into, args.sources, progress=show)
    except ValueError as bad:
        print(f"\n{bad}", file=sys.stderr)
        return 1
    print(file=sys.stderr)
    print(f"restored {report.books:,} books and {report.chunks:,} chunks into {report.data_dir}")
    if report.relocated:
        print(f"  {report.relocated:,} source paths pointed at {report.library}")
    if report.missing:
        print(f"\n{len(report.missing)}+ sources are not readable, for example:", file=sys.stderr)
        for path in report.missing[:3]:
            print(f"  {path}", file=sys.stderr)
        print("search will rank but cannot show passages; `dyp check` lists them all.",
              file=sys.stderr)
        return 1
    print("  every source is readable; nothing needs re-embedding")

    weights = manifest.get("weights") or []
    if weights:
        print("\nthese vectors were made with:")
        for w in weights:
            print(f"  {w['name']}")
            if w.get("file_name"):
                size = f"  {_size(w['file_bytes'])}" if w.get("file_bytes") else ""
                print(f"    file    {w['file_name']}{size}")
            if w.get("file_sha256"):
                print(f"    sha256  {w['file_sha256']}")
            if w.get("source_uri"):
                print(f"    from    {w['source_uri']}")
        print("\nif you already have those weights, point --model at them and every "
              "vector here is reused.")
        print("check a candidate first with:  dyp -L NAME models --verify PATH.gguf")
    if args.register:
        from dyprys import registry
        entry = registry.add(args.register, report.data_dir)
        print(f"\nregistered as {entry.name}"
              f"{'  (default)' if entry.is_default else ''} — use it with -L {entry.name}")
    return 0


def _relocate(conn, args) -> int:
    """Point the index at the library's new location."""
    from dyprys.library import relocate, unreadable_sources

    books, sources = relocate(conn, args.old, args.new)
    if not sources:
        print(f"nothing recorded under {args.old!r}", file=sys.stderr)
        print("run `dyp books` to see the prefixes actually stored.", file=sys.stderr)
        return 1

    print(f"rewrote {sources:,} source path(s) and {books:,} book key(s)")
    missing = unreadable_sources(conn)
    if missing:
        print(f"\nbut {len(missing)}+ are still not on disk, for example:", file=sys.stderr)
        for path in missing[:3]:
            print(f"  {path}", file=sys.stderr)
        print("check the new prefix; `dyp check` lists them all.", file=sys.stderr)
        return 1
    print("every source is readable at the new location; vectors are untouched")
    return 0


def _add(conn, args) -> int:
    from dyprys.ingest import NothingToIngest

    suffixes = tuple(
        e if e.startswith(".") else f".{e}"
        for e in (x.strip().lower() for x in args.ext.split(",")) if e
    )
    # Named files that the filter will drop. Reported rather than passed over:
    # a glob that quietly ingested half of what it matched would be worse than
    # one that ingested none of it.
    from dyprys.ingest import acceptable
    skipped = [p for p in args.paths if p.is_file() and not acceptable(p, suffixes)]

    try:
        results = ingest_paths(
            conn,
            args.paths,
            target=args.target,
            overlap=args.overlap,
            chapters=args.chapters,
            deep=args.deep,
            suffixes=suffixes,
        )
    except NothingToIngest as nothing:
        wanted = ", ".join(nothing.wanted)
        print(f"no {wanted} files found under: "
              f"{', '.join(str(r) for r in nothing.roots)}", file=sys.stderr)
        instead = nothing.found_instead()
        if instead:
            listing = ", ".join(f"{ext} ({n})" for ext, n in instead)
            print(f"what is there instead: {listing}", file=sys.stderr)
            print(f"pass --ext to include them, e.g. --ext {instead[0][0]}", file=sys.stderr)
        else:
            print("that directory holds no files with an extension at all.", file=sys.stderr)
        return 1
    if not results:
        print("no text files found", file=sys.stderr)
        return 1

    if skipped:
        kinds = sorted({p.suffix.lower() for p in skipped})
        print(f"skipped {len(skipped)} file(s) of a kind this build does not read: "
              f"{', '.join(kinds)}", file=sys.stderr)
        print(f"pass --ext to include them, e.g. --ext {kinds[0]}", file=sys.stderr)

    counts: dict[str, int] = {}
    carried = 0
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        carried += r.carried
        if r.status != "unchanged":
            sources = f"{r.sources} sources" if r.sources > 1 else "1 source"
            note = f"  ({r.carried:,} vectors kept)" if r.carried else ""
            print(f"{r.status:>9}  {r.chunks:>6,} chunks  {sources:>11}  {r.title}{note}")

    summary = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    print(f"\n{len(results)} books: {summary}")
    if any(r.status != "unchanged" for r in results):
        db.record_event(conn, "add", f"{len(results)} books: {summary}")

    # The same book from two extractions has different bytes, so the content
    # hash cannot see it. A shared title can, and the cost of missing it is a
    # second copy of the embedding bill.
    from dyprys.library import duplicate_titles
    dupes = duplicate_titles(conn)
    if dupes:
        total = sum(n - 1 for _, n in dupes)
        print(f"\n{total} book(s) share a title with another already indexed:",
              file=sys.stderr)
        for title, n in dupes[:5]:
            print(f"  x{n}  {title[:66]}", file=sys.stderr)
        if len(dupes) > 5:
            print(f"  … and {len(dupes) - 5} more titles", file=sys.stderr)
        print("two extractions of one book hash differently, so this is the only "
              "way to notice.", file=sys.stderr)
        print("`dyp books PATTERN` to compare them; `dyp remove` if it is a "
              "duplicate.", file=sys.stderr)

    # Cheapest possible moment to learn a PDF extracted as gibberish: before any
    # GPU has been spent on it, while re-extracting is still just a re-run.
    from dyprys.check import garbled_books
    fresh = {row["id"] for row in conn.execute(
        "SELECT id FROM books WHERE key IN (%s)" % ",".join("?" * len(results)),
        [str(r.key) for r in results])} if results else set()
    broken = garbled_books(conn, only=fresh)
    if broken:
        chunks = sum(b.chunks for b in broken)
        print(f"\n{len(broken)} book(s) have no word boundaries — {chunks:,} chunks:",
              file=sys.stderr)
        for b in broken[:5]:
            print(f"  {b.chunks:>6,} chunks, longest words ~{b.p90_token}  {b.title[:52]}",
                  file=sys.stderr)
        print("the PDF's font map likely defeated the extractor. These embed at full "
              "cost and", file=sys.stderr)
        print("can never match a query — re-extract before `dyp embed`, or "
              "`dyp remove` them.", file=sys.stderr)
    if carried:
        print(f"{carried:,} chunks unchanged by the edit — their vectors are reusable")
    _say_next(conn)
    return 0


def _embed(conn, directory, args) -> int:
    """Resumable. Interrupting it is normal and loses at most one batch."""
    # Resolved before the weights are opened: a pattern matching nothing should
    # cost milliseconds, not a 334 MB model load followed by an empty run. And
    # it must never widen to the whole library -- the same rule `ask -c`
    # follows, but the mistake is far more expensive here, measured in GPU-hours.
    scope = None
    if getattr(args, "collection", None):
        from dyprys.search import scope_books

        scope = scope_books(conn, args.collection)
        if not scope:
            print(f"no book matches {args.collection!r}", file=sys.stderr)
            return 1
        total_books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        print(f"scoped to {len(scope):,} book(s) of {total_books:,} "
              f"matching {args.collection!r}", file=sys.stderr)

    # Which way of splitting this model works on. A model embeds one chunking
    # and only one, so this is settled before any weights are opened.
    ways = db.chunkings(conn)
    chosen = None
    if getattr(args, "target", None):
        matching = [c for c in ways if c["target"] == args.target]
        if not matching:
            print(f"no chunking with target {args.target:,} bytes. This library has: "
                  + ", ".join(f"{c['target']:,}" for c in ways), file=sys.stderr)
            return 1
        chosen = matching[0]["id"]
    elif len(ways) == 1:
        chosen = ways[0]["id"]
    elif len(ways) > 1:
        print("this library is split more than one way; say which to embed:",
              file=sys.stderr)
        for c in ways:
            print(f"  --target {c['target']:<7,} {c['chunks']:>10,} chunks",
                  file=sys.stderr)
        return 1

    model_path, why = _weights_for(conn, args.model)
    if model_path is None:
        print(why, file=sys.stderr)
        return 2

    from dyprys.embed import embed_pending, store_for
    from dyprys.embedder import Embedder
    from dyprys.lock import AlreadyRunning, exclusive

    print(f"loading {Path(model_path).name} …", file=sys.stderr)
    embedder = Embedder(model_path)
    model_id = db.model_id(
        conn, embedder.name, embedder.dim, "int8" if args.int8 else "fp32",
        provenance=embedder.provenance(),
    )
    db.remember_weights(conn, model_id, Path(model_path).resolve())
    if chosen is not None:
        try:
            db.bind_chunking(conn, model_id, chosen)
        except ValueError as clash:
            print(clash, file=sys.stderr)
            return 1
    chosen = db.bound_chunking(conn, model_id) or chosen
    # A restored index carries vectors under the identity of the weights that
    # made them. If this file is not those weights, none of that work applies --
    # and the only symptom would be a very long run that looks normal.
    others = conn.execute(
        "SELECT m.name, COALESCE(SUM(p.n_embedded), 0) AS done FROM models m "
        "LEFT JOIN segment_progress p ON p.model_id = m.id "
        "WHERE m.id != ? GROUP BY m.id HAVING done > 0",
        (model_id,),
    ).fetchall()
    mine = conn.execute(
        "SELECT COALESCE(SUM(n_embedded), 0) FROM segment_progress WHERE model_id = ?",
        (model_id,),
    ).fetchone()[0]
    if others and not mine:
        print(f"\nthis index already holds vectors, but under other weights:", file=sys.stderr)
        for row in others:
            print(f"  {row['name']}  ({row['done']:,} chunks)", file=sys.stderr)
        print(f"yours is {embedder.name}, which has nothing embedded yet, so this "
              f"run starts from zero.", file=sys.stderr)
        print("if you meant to use the existing vectors, point --model at the file "
              "they were made with.\n", file=sys.stderr)

    stored_as = db.model_quantisation(conn, model_id)
    if args.int8 and stored_as != "int8":
        print(f"{embedder.name} is already registered as {stored_as}; "
              f"run `dyp models --quantise` to convert it", file=sys.stderr)
        return 2
    store = store_for(conn, directory, model_id, embedder.dim)

    # The same clock the deadline uses. time.time() counts the hours a laptop
    # spends asleep, and `--for` does not: a 30-minute run on a Mac that idled
    # reported "1h19m elapsed, 1.4 chunks/s" for 29.5 minutes of work at 5.9/s.
    # Reporting a rate that includes sleep also poisons the estimate of what is
    # left, which is the number the flag exists to make trustworthy.
    started = time.monotonic()
    # Wall clock over the same span, kept only so its difference from the
    # monotonic one can be recorded. That difference is how much of the run the
    # machine spent asleep, and it is not recoverable afterwards from either
    # clock alone.
    started_wall = time.time()
    began = db.now()

    def show(seen, total, book):
        rate = seen / max(time.monotonic() - started, 1e-9)
        left = (total - seen) / rate if rate else 0
        title = book if len(book) <= 34 else book[:33] + "…"
        print(
            f"\r  {seen:,}/{total:,}  {seen/total:4.0%}  {rate:4.1f}/s  "
            f"{_clock(time.monotonic() - started)} elapsed, {_clock(left)} left  {title:<35}",
            end="", file=sys.stderr, flush=True,
        )

    # Ctrl-C asks the loop to stop after the batch in flight, so the work already
    # paid for is committed rather than discarded. A second Ctrl-C is taken as
    # "I meant now" and lets the default handler through.
    stopping = {"asked": False}

    def on_interrupt(signum, frame):
        if stopping["asked"]:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        stopping["asked"] = True
        print("\n  finishing the current batch, then stopping "
              "(Ctrl-C again to stop now) …", file=sys.stderr, flush=True)

    previous = signal.signal(signal.SIGINT, on_interrupt)
    try:
        with exclusive(directory, "embed"):

            report = embed_pending(
                conn, store, embedder, model_id,
                limit=args.limit, batch_size=args.batch,
                seconds=_duration(args.duration),
                should_stop=lambda: stopping["asked"],
                progress=show,
                book_ids=scope,
                chunking_id=chosen,
                duty=max(1, min(100, args.duty)) / 100,
            )
    except AlreadyRunning as busy:
        print(f"\n{busy}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        store.flush()
        print("\nstopped immediately — the batch in flight was discarded, "
              "everything before it is committed", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGINT, previous)
        store.close()

    print(file=sys.stderr)
    elapsed = time.monotonic() - started          # same clock as `started`
    db.record_run(conn, model_id, began, elapsed, report, time.time() - started_wall)
    if report.stopped != "complete":
        reason = {"time": "the time limit", "limit": "the chunk limit",
                  "interrupted": "your request"}[report.stopped]
        print(f"stopped at {reason}; everything embedded is committed.")
        print("run `dyp embed` again to pick up exactly where this left off.")
    print(f"embedded   {report.embedded:>10,}  in {elapsed:.1f}s")
    if report.copied:
        print(f"copied     {report.copied:>10,}  carried across an edit")
    if report.failed:
        print(f"failed     {report.failed:>10,}  recorded, retryable")
    if report.embedded:
        print(f"rate       {report.embedded / elapsed:>10.1f}  chunks/s")

    # A scoped run that finishes its scope looks exactly like a run that
    # finished the library, and `_say_next` would then advise routing as though
    # nothing were outstanding. Say what was left out, and how to reach it.
    if scope is not None:
        from dyprys.embed import _outstanding

        rest = _outstanding(conn, model_id, None, chosen) - _outstanding(
            conn, model_id, scope, chosen)
        if rest > 0:
            print(f"\n{rest:,} chunk(s) outside {args.collection!r} are still "
                  f"unembedded.")
            print(f"`dyp embed` for all of them, or `dyp embed -c PATTERN` for "
                  f"another part.")
            return 0
    _say_next(conn)
    return 0


def _clock(seconds: float) -> str:
    """h:mm, or m:ss under an hour — a number a person can act on."""
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _where(args) -> Path:
    """Which index this command acts on.

    An explicit path wins, then a name given on the command line, then whichever
    library was made the default, and only then the ./data fallback. Explicit
    always beats remembered, so a stale default can never silently redirect a
    command that named its target.
    """
    from dyprys import registry

    if args.data:
        return Path(args.data)
    if getattr(args, "library", None):
        found = registry.resolve(args.library)
        if found is None:
            if not registry.readable():
                raise SystemExit(
                    f"the registry at {registry.registry_path()} could not be read, "
                    f"so no name resolves. Open the index directly with --data DIR."
                )
            raise SystemExit(f"no library named {args.library!r}; try `dyp library list`")
        return found
    if not os.environ.get("DYPRYS_DATA"):
        default = registry.resolve(None)
        if default is not None:
            return default
    return db.data_dir(None)


def _sources_under(index_dir: Path) -> str | None:
    """Where this index's text lives, so a delete can say what it is sparing."""
    try:
        conn = db.connect(index_dir)
    except Exception:
        return None
    try:
        row = conn.execute("SELECT path FROM sources LIMIT 1").fetchone()
        return str(Path(row["path"]).parent) if row else None
    finally:
        conn.close()


def _library(args, parser) -> int:
    """Name the libraries this installation knows about."""
    from dyprys import registry

    action = getattr(args, "action", None)
    # An unreadable registry looks exactly like an empty one, and saying "no
    # libraries registered yet" to someone who registered five is the message
    # that turns a recoverable file into a lost one.
    damaged = not registry.readable()
    if damaged:
        print(f"the registry at {registry.registry_path()} exists but could not be "
              f"read.", file=sys.stderr)
        print("its names are unavailable; every library still opens with `--data DIR`.",
              file=sys.stderr)
        if action in ("add", "remove", "use"):
            print(f"writing now keeps it as *{registry.UNREADABLE_SUFFIX} rather than "
                  f"overwriting it — the names in it are recoverable by hand.",
                  file=sys.stderr)
        print(file=sys.stderr)

    if action == "add":
        entry = registry.add(args.name, args.path)
        marker = "  (default)" if entry.is_default else ""
        print(f"{entry.name} → {entry.path}{marker}")
        if not entry.exists:
            print("  no index there yet — `dyp -L %s add ...` will create one" % entry.name,
                  file=sys.stderr)
        return 0
    if action == "remove":
        entry = next((l for l in registry.libraries() if l.name == args.name), None)
        if entry is None:
            print(f"no library named {args.name!r}", file=sys.stderr)
            return 1

        if args.delete:
            files, size = registry.contents(entry.path)
            print(f"this would delete {entry.path}")
            print(f"  {files} file(s), {_size(size)} — the index, its vectors and its "
                  f"routing profile")
            sources = _sources_under(entry.path)
            if sources:
                print(f"  the text itself is elsewhere and is NOT touched, e.g. {sources}")
            if not args.yes:
                print("\nre-run with --yes to go ahead.", file=sys.stderr)
                return 1
            import shutil
            shutil.rmtree(entry.path, ignore_errors=True)
            registry.remove(args.name)
            print(f"deleted {entry.path} and forgot {args.name}")
            return 0

        registry.remove(args.name)
        print(f"forgot {args.name}; its files are untouched at {entry.path}")
        print(f"delete them too with: dyp library remove {args.name} --delete --yes")
        return 0
    if action == "use":
        if not registry.use(args.name):
            print(f"no library named {args.name!r}", file=sys.stderr)
            return 1
        print(f"{args.name} is now the default")
        return 0

    entries = registry.libraries()

    if getattr(args, "json", False):
        libs = []
        for e in entries:
            row = {"name": e.name, "path": str(e.path), "default": e.is_default,
                   "exists": e.exists}
            held = registry.summarise(e.path) if e.exists else None
            if held is not None:
                row.update(books=held.books, chunks=held.chunks, models=[
                    {"name": name, "embedded": done, "total": whole,
                     "coverage": (done / whole) if whole else 0.0}
                    for name, done, whole in held.models])
            libs.append(row)
        return _emit_json({
            "libraries": libs,
            "registry_path": str(registry.registry_path()),
            "registry_readable": not damaged,
        })

    if not entries:
        if damaged:
            return 1              # already explained, and it is not "yet"
        print("no libraries registered yet — `dyp library add NAME DIR`", file=sys.stderr)
        print(f"the registry lives at {registry.registry_path()}", file=sys.stderr)
        return 1
    width = max(len(e.name) for e in entries)
    print(f"  {'name':<{width}}  {'books':>7} {'chunks':>10} {'embedded':>10} {'':>6}  model")
    for entry in entries:
        mark = "*" if entry.is_default else " "
        if not entry.exists:
            print(f"{mark} {entry.name:<{width}}  {'—':>7} {'—':>10} {'—':>10} {'':>6}  "
                  f"(no index yet)  {entry.path}")
            continue
        held = registry.summarise(entry.path)
        if held is None:
            print(f"{mark} {entry.name:<{width}}  {'unreadable':>7}  {entry.path}")
            continue
        if not held.models:
            print(f"{mark} {entry.name:<{width}}  {held.books:>7,} {held.chunks:>10,} "
                  f"{'—':>10} {'':>6}  no model yet")
        for i, (name, done, whole) in enumerate(held.models):
            # Against what this model could embed, not the library. Same
            # per-model-over-library-wide mistake as `dyp watch` and `dyp models`.
            share = done / whole if whole else 0.0
            head = f"{mark} {entry.name:<{width}}" if i == 0 else f"  {'':<{width}}"
            counts = (f"{held.books:>7,} {held.chunks:>10,}" if i == 0
                      else f"{'':>7} {'':>10}")
            print(f"{head}  {counts} {done:>10,} {share:>6.0%}  {_shorten(name)}")
    print(f"\n* is the default. {registry.registry_path()}")
    print("paths: " + ", ".join(f"{e.name}={e.path}" for e in entries))
    return 0


def _ago(stamp: str) -> str:
    """How long ago, in the coarsest unit that is still informative.

    "3m ago" is what you want when asking what just happened; an ISO timestamp
    is what you want when asking what happened on the 14th. Both are shown.
    """
    from datetime import datetime, timezone

    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return ""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - when).total_seconds()
    if seconds < 0:
        return "just now"
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return f"{int(seconds)}s ago"


def next_step(conn) -> str | None:
    """The one command that most usefully comes next, or None when idle.

    A library has a lifecycle -- add, embed, route, ask -- and until now nothing
    said so. `dyp add` ended on "2 books: 2 added" and left a new user with no
    idea that embedding was a separate step, let alone that routing was a third.
    The information was always available from `dyp check`; the problem was that
    you had to know to look.

    Deliberately one line and one command. A checklist of four things to do next
    is a different kind of unhelpful.
    """
    from dyprys.check import survey
    from dyprys.routing import is_built, stale_books

    report = survey(conn)
    if not report.live_chunks:
        return "add some text — `dyp add PATH`"
    if not report.models:
        return "embed it — `dyp embed --model PATH.gguf` (remembered after the first run)"

    model = max(report.models, key=lambda m: m.embedded)
    if model.to_embed or model.to_copy:
        outstanding = model.to_embed + model.to_copy
        return f"embed the remaining {outstanding:,} chunk(s) — `dyp embed`"
    row = conn.execute("SELECT id FROM models ORDER BY id").fetchone()
    model_id = row["id"] if row else None
    if model_id is not None:
        if not is_built(conn, model_id):
            return "profile the books so search can skip most of them — `dyp route`"
        if stale_books(conn, model_id):
            return "the routing profile is out of date — `dyp route`"
    if report.lexical_chunks < report.live_chunks:
        return "build the keyword index — `dyp lexical`"
    return None


def _say_next(conn) -> None:
    """Print the next step, if there is one.  Never fatal."""
    try:
        step = next_step(conn)
    except Exception:
        return              # a hint must never be the reason a command fails
    if step:
        print(f"\nnext: {step}")


def _local(stamp: str) -> str:
    """A stored timestamp as wall-clock time where the reader is sitting.

    Timestamps are stored UTC, which is right -- a library backed up in one zone
    and restored in another must not shift. Rendering them is the other half,
    and slicing the ISO string was not it: `[:19]` drops the `+00:00` and prints
    UTC digits under a local-looking heading. Reconciling the journal against a
    shell log caught it, an hour out, on a machine on BST -- and the whole point
    of the journal is to say when something happened.
    """
    from datetime import datetime, timezone

    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp[:19].replace("T", " ")
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone().strftime("%Y-%m-%d %H:%M:%S")


# A run's two clocks never agree to the second; only a gap worth reading is one
# the machine actually slept through.
SLEPT_AT_LEAST = 60


def _asleep(run) -> float:
    """Seconds this run spent with the machine asleep, or 0 if not knowable.

    `seconds` is monotonic and stops while the machine sleeps; `wall_seconds`
    does not. Their difference is the sleep, and it is the only way to tell a
    run that worked for thirty minutes from one that was begun thirty minutes of
    work ago -- which on a laptop are routinely hours apart.
    """
    try:
        wall = run["wall_seconds"]
    except (IndexError, KeyError):
        return 0.0
    if wall is None:
        return 0.0
    gap = wall - run["seconds"]
    return gap if gap >= SLEPT_AT_LEAST else 0.0


def _asked(conn, args) -> int:
    """What was asked before, and what it produced."""
    import json

    def as_row(row):
        return {"id": row["id"], "at": row["at"], "question": row["question"],
                "mode": row["mode"], "ms": row["ms"],
                "detail": json.loads(row["detail"])}

    if getattr(args, "json", False) and not args.forget:
        # A read of the question log, so --forget (which writes) is not a --json
        # operation. One question if numbered, else the list.
        allrows = db.questions_asked(conn, limit=10 ** 9)
        if args.which is not None:
            one = next((r for r in allrows if r["id"] == args.which), None)
            return _emit_json(as_row(one)) if one else (_emit_json(None) or 1)
        rows = db.questions_asked(conn, args.limit, args.find)
        return _emit_json({"questions": [as_row(r) for r in rows]})

    if args.forget:
        what = args.forget.lower()
        if what == "all":
            doomed, before, which = db.questions_asked(conn, limit=10 ** 9), None, None
        elif what.isdigit():
            which, before = int(what), None
            doomed = [r for r in db.questions_asked(conn, limit=10 ** 9) if r["id"] == which]
        else:
            before, which = args.forget, None
            doomed = [r for r in db.questions_asked(conn, limit=10 ** 9)
                      if r["at"] < args.forget]
        if not doomed:
            print("nothing matches", file=sys.stderr)
            return 1
        print(f"this would forget {len(doomed)} question(s):")
        for row in doomed[:5]:
            print(f"  {_local(row['at'])}  {row['question'][:60]}")
        if len(doomed) > 5:
            print(f"  … and {len(doomed) - 5} more")
        if not args.yes:
            print("\nre-run with --yes to go ahead.", file=sys.stderr)
            return 1
        gone = db.forget_questions(conn, before=before, which=which)
        print(f"\nforgot {gone} question(s)")
        return 0

    if args.which is not None:
        rows = [r for r in db.questions_asked(conn, limit=10 ** 9) if r["id"] == args.which]
        if not rows:
            print(f"no question numbered {args.which}", file=sys.stderr)
            return 1
        row = rows[0]
        detail = json.loads(row["detail"])
        print(f"{term.bold(row['question'])}")
        print(term.dim(f"  {_local(row['at'])} · {row['mode']} · {row['ms']:.0f} ms"
                       + (f" · {detail['books']} routed books" if detail.get("books") else "")))
        for role, name in (detail.get("models") or {}).items():
            print(term.dim(f"  {role}: {name}"))
        for kind, lines in (detail.get("expansion") or {}).items():
            for line in lines:
                print(term.dim(f"  expanded ({kind}): ") + line[:100])
        print()
        for rank, hit in enumerate(detail.get("hits") or [], 1):
            print(f"{term.bold(str(rank) + '.')} "
                  f"{term.dim('[cos %.2f · %s]' % (hit['cos'], hit['why'] or '—'))} "
                  f"{term.bold(hit['title'])}")
            print(term.dim(f"   {_where_to_look(hit['path'], hit['offset'])}"))
        answer = detail.get("answer")
        if answer:
            print("\n" + term.rule())
            if detail.get("models", {}).get("retried"):
                print(term.dim("searched again for: ") + detail["models"]["retried"])
            print(answer["prose"])
            for claim in answer.get("verified", []):
                print(term.style("  ✓", "green") + f" [{claim['cited']}] {claim['where']}")
            for claim in answer.get("rejected", []):
                print(term.style("  ✗", "red") + f" [{claim['cited']}] {claim['quote'][:60]}")
        return 0

    rows = db.questions_asked(conn, args.limit, args.find)
    if not rows:
        print("no questions recorded yet — `dyp ask` keeps each one", file=sys.stderr)
        return 1
    print(term.dim(f"{'#':>5}  {'when':<20}{'ms':>8}   question"))
    for row in reversed(rows):
        detail = json.loads(row["detail"])
        marks = "".join((
            "r" if detail.get("routed") else "",
            "e" if detail.get("expansion") else "",
            "s" if detail.get("answer") else "",
        ))
        # Padded before styling. An escape code has no width on screen and full
        # width to str.format, so styling first makes every column ragged — the
        # same mistake as wrapping text after marking it.
        number = f"{row['id']:>5}"
        when = f"{_local(row['at']):<20}"
        ms = f"{row['ms']:>8.0f}"
        print(f"{term.dim(number)}  {term.dim(when)}{term.dim(ms)}   "
              f"{row['question'][:56]} {term.dim(marks)}")
    print(term.dim("\n  r routed · e expanded · s summarised"))
    print(term.dim(f"  `dyp asked N` for one in full · `dyp asked --forget all`"))
    return 0


def _history(conn, limit: int, as_json: bool = False) -> int:
    """What has been done to this index, and what embedding it cost."""
    events = conn.execute(
        "SELECT at, action, detail FROM events ORDER BY id DESC LIMIT ?",
        (limit,)).fetchall()
    runs = conn.execute(
        "SELECT r.*, m.name FROM embed_runs r JOIN models m ON m.id = r.model_id "
        "ORDER BY r.id DESC LIMIT ?", (limit,)).fetchall()

    if as_json:
        total = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(seconds),0), COALESCE(SUM(embedded),0) "
            "FROM embed_runs").fetchone()
        return _emit_json({
            "events": [
                {"at": e["at"], "action": e["action"], "detail": e["detail"]}
                for e in reversed(events)],
            "runs": [
                {"started_at": r["started_at"], "seconds": r["seconds"],
                 "wall_seconds": (r["wall_seconds"] if "wall_seconds" in r.keys() else None),
                 "embedded": r["embedded"], "copied": r["copied"], "failed": r["failed"],
                 "rate": (r["embedded"] / r["seconds"]) if r["seconds"] else 0.0,
                 "stopped": r["stopped"], "model": r["name"],
                 "asleep_seconds": _asleep(r)}
                for r in reversed(runs)],
            "totals": {"runs": total[0], "seconds": total[1], "chunks": total[2],
                       "mean_rate": (total[2] / total[1]) if total[1] else 0.0},
        })

    if events:
        print(f"what has been done to this index (most recent {len(events)})")
        for e in reversed(events):
            print(f"  {_local(e['at']):<20} {_ago(e['at']):>9}  {e['action']:<9} {e['detail']}")
        print()

    rows = runs
    if not rows:
        sys.stdout.flush()          # or the note lands above the events
        print("no embedding runs recorded yet — `dyp embed` records each one when it stops.",
              file=sys.stderr)
        return 0 if events else 1

    print("embedding runs")
    print(f"{'started':<20} {'ran for':>9} {'embedded':>10} {'rate':>9} {'stopped':>12}")
    for r in reversed(rows):
        rate = r["embedded"] / r["seconds"] if r["seconds"] else 0.0
        extra = ""
        if r["copied"]:
            extra += f"  +{r['copied']:,} carried"
        if r["failed"]:
            extra += f"  {r['failed']:,} failed"
        asleep = _asleep(r)
        if asleep:
            extra += f"  {_clock(asleep)} asleep"
        print(f"{_local(r['started_at']):<20} {_clock(r['seconds']):>9} "
              f"{r['embedded']:>10,} {rate:>7.1f}/s {r['stopped']:>12}{extra}")

    total = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(seconds),0), COALESCE(SUM(embedded),0) "
        "FROM embed_runs").fetchone()
    print(f"\n{total[0]:,} runs, {_clock(total[1])} of embedding, {total[2]:,} chunks")
    if total[1]:
        print(f"  mean {total[2] / total[1]:.1f}/s over everything")
    for m in conn.execute("SELECT id, name FROM models ORDER BY id"):
        rate = db.observed_rate(conn, m["id"])
        if rate:
            print(f"  median {rate:.1f}/s for {_shorten(m['name'])}"
                  f"  — what `dyp check` estimates from")
    return 0


def _model_being_embedded(conn):
    """The model a running embed is most likely working on.

    The lock records a pid, not a model, so this is inferred: the model with
    work left in the chunking it is bound to. With one model that is trivially
    the right answer; with two it picks the unfinished one, which is the only
    one an embed could be running for.
    """
    best = None
    for m in conn.execute("SELECT id, name, chunking_id FROM models ORDER BY id"):
        if m["chunking_id"] is None:
            live = conn.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM segments").fetchone()[0]
        else:
            live = conn.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM segments WHERE chunking_id = ?",
                (m["chunking_id"],)).fetchone()[0]
        done = conn.execute(
            "SELECT COALESCE(SUM(n_embedded), 0) FROM segment_progress WHERE model_id = ?",
            (m["id"],)).fetchone()[0]
        left = live - done
        if left > 0 and (best is None or left > best[0]):
            best = (left, m["id"], m["name"], m["chunking_id"])
    return None if best is None else (best[1], best[2], best[3])


def _watch(conn, directory, every: float) -> int:
    """Follow an embed started elsewhere, by reading what it commits.

    Not coupled to the running process at all: progress is committed per batch,
    so any reader sees it advance. That is why this works from another terminal,
    over ssh, or after the terminal that started the job has gone.
    """
    from dyprys.lock import holder

    pid = holder(directory, "embed")
    if pid is None:
        print("no embedding run in progress here.", file=sys.stderr)
        print("`dyp check` for what is outstanding; `dyp embed` to start one.",
              file=sys.stderr)
        return 1
    who = f"pid {pid}" if pid > 0 else "pid unknown"
    print(f"following the embed holding this index ({who}) — Ctrl-C to stop watching\n",
          file=sys.stderr)

    # Which model is being embedded. Summing progress across all of them counted
    # a finished model's work toward a running one's total, so a second model
    # embedding into a library the first had completed showed over 100% done.
    following = _model_being_embedded(conn)
    if following is None:
        print("nothing outstanding for any model here.", file=sys.stderr)
        return 1
    model_id, model_name, chunking_id = following
    if conn.execute("SELECT COUNT(*) FROM models").fetchone()[0] > 1:
        print(f"following {_shorten(model_name)}\n", file=sys.stderr)

    def totals():
        if chunking_id is None:
            live = conn.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM segments").fetchone()[0]
        else:
            live = conn.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM segments "
                "WHERE chunking_id = ?", (chunking_id,)).fetchone()[0]
        done = conn.execute(
            "SELECT COALESCE(SUM(n_embedded), 0) FROM segment_progress "
            "WHERE model_id = ?", (model_id,)).fetchone()[0]
        row = conn.execute(
            "SELECT b.title, p.n_embedded, seg.chunk_count FROM segment_progress p "
            "JOIN segments seg ON seg.id = p.segment_id "
            "JOIN sources src ON src.id = seg.source_id "
            "JOIN books b ON b.id = src.book_id "
            "WHERE p.model_id = ? AND p.n_embedded > 0 "
            "AND p.n_embedded < seg.chunk_count LIMIT 1", (model_id,)).fetchone()
        return live, done, row

    live_display = _stderr_is_tty()
    painted = False
    first_done, started = None, time.monotonic()
    try:
        while True:
            live, done, inflight = totals()
            if first_done is None:
                first_done = done
            elapsed = time.monotonic() - started
            rate = (done - first_done) / elapsed if elapsed > 1 else 0.0
            left = (live - done) / rate if rate > 0 else None
            title = inflight["title"] if inflight else "—"
            book = term.elide(title, 38)
            eta = _clock(left) if left else "  --  "
            b_done = inflight["n_embedded"] if inflight else 0
            b_all = inflight["chunk_count"] if inflight else 0
            b_share = f"{b_done / b_all:4.0%}" if b_all else "  --"

            whole = (f"  {term.bar(done, live)}  {done / live:5.1%}  "
                     f"{done:>9,}/{live:,}  {rate:5.1f}/s  {eta} left")
            this = (f"  {term.bar(b_done, b_all)}  {b_share}  "
                    f"{term.dim(book)}")
            if live_display:
                # Redrawn in place, and the cursor is left on the line *below*
                # rather than blinking inside the display. Each pass goes back
                # up two lines, rewrites both, and ends one line further down.
                up = "\033[2A" if painted else ""
                print(f"{up}\r\033[K{whole}\n\033[K{this}\n\033[K",
                      end="", file=sys.stderr, flush=True)
                painted = True
            else:
                print(f"{whole}\n{this}", file=sys.stderr, flush=True)
            if holder(directory, "embed") is None:
                print("\nthe run finished or stopped.", file=sys.stderr)
                return 0
            time.sleep(max(0.2, every))
    except KeyboardInterrupt:
        print("\nstopped watching. The run is untouched — this only reads.",
              file=sys.stderr)
        return 0


def _duration(text: str | None) -> float | None:
    """`90` seconds, or `30m`, or `2h`."""
    if not text:
        return None
    units = {"s": 1, "m": 60, "h": 3600}
    if text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def _weights_for(conn, model_arg):
    """Where to load weights from: a path, a remembered model, or nothing.

    The index already records the file name, size and digest of the weights that
    made its vectors, and now where they were last opened. So the common case --
    one model, still on this machine -- needs no argument at all.
    """
    given = model_arg or os.environ.get("DYPRYS_MODEL")
    if given and Path(given).exists():
        return given, None

    row = db.find_model(conn, given) if given else db.sole_model(conn)
    if row is None:
        if given:
            return None, (f"no model matches {given!r}, and it is not a file. "
                          f"`dyp models` lists what this index knows.")
        registered = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        if registered > 1:
            return None, ("this index has several models — say which with "
                          "--model NAME. `dyp models` lists them.")
        return None, ("no model yet: pass --model PATH.gguf or set $DYPRYS_MODEL. "
                      "It is remembered after the first run.")

    remembered = row["file_path"]
    if remembered and Path(remembered).exists():
        # A hint, not a promise. Size is the cheap half of the check; the digest
        # is the real one and happens on load, where wrong weights register as a
        # different model rather than quietly polluting this one.
        size = Path(remembered).stat().st_size
        if row["file_bytes"] and size != row["file_bytes"]:
            return None, (f"the weights remembered at {remembered} are "
                          f"{_size(size)}, not the {_size(row['file_bytes'])} that "
                          f"made these vectors. Pass --model explicitly.")
        return remembered, None
    where = f" (last seen at {remembered})" if remembered else ""
    return None, (f"this index needs {row['file_name'] or row['name']}{where}, "
                  f"which is not on this machine now.\n"
                  f"pass --model PATH.gguf; `dyp models --needed` shows how to "
                  f"verify a candidate.")


def _load_model(conn, directory, model_arg):
    """Open the model named on the command line and its vector file."""
    path, why = _weights_for(conn, model_arg)
    if path is None:
        print(why, file=sys.stderr)
        return None
    from dyprys.embed import store_for
    from dyprys.embedder import Embedder

    embedder = Embedder(path)
    # An index whose vectors were truncated needs its queries truncated to
    # match. Without this the model row says 512, the embedder says 768, and
    # every command refuses to open the index it just shrank.
    known = conn.execute("SELECT dim FROM models WHERE name = ?",
                         (embedder.name,)).fetchone()
    if known and known["dim"] < embedder.dim:
        from dyprys.embedder import Truncated

        embedder = Truncated(embedder, known["dim"])
    model_id = db.model_id(
        conn, embedder.name, embedder.dim, provenance=embedder.provenance()
    )
    db.remember_weights(conn, model_id, Path(path).resolve())
    return embedder, model_id, store_for(conn, directory, model_id, embedder.dim)


def results_as_json(question, mode, routed, scanned, elapsed_ms, passages, why, cosine):
    """The `--json` payload, built from resolved passages and nothing else.

    Kept a pure function so it can be tested without a model or an index: it turns
    what a search already produced into the shape an agent parses. `text` is null
    when the source could not be proved, and `state` says why; `provenance` is the
    tool's own rank signal ("vec 1 · phrase 1"), which carries more than a single
    fused score could.
    """
    results = []
    for rank, p in enumerate(passages, 1):
        results.append({
            "rank": rank,
            "chunk_id": p.chunk_id,
            "book": p.title,
            "chapter": (p.chapter + 1) if p.chapter else None,
            "path": str(p.path),
            "offset": p.offset,
            "cos": round(cosine[p.chunk_id], 4) if p.chunk_id in cosine else None,
            "provenance": why.get(p.chunk_id),
            "state": p.state,
            "text": p.text,
        })
    return {
        "query": question,
        "mode": mode,
        "routed": routed,
        "scanned_fraction": round(scanned, 4),
        "elapsed_ms": round(elapsed_ms, 1),
        "results": results,
    }


def _ask(conn, directory, args) -> int:
    # An empty or whitespace query embeds to a meaningless vector and matches no
    # words, so hybrid search returns whatever the vector half drifts to -- noise
    # presented as answers, at exit 0. Refuse it instead.
    args.question = args.question.strip()
    if not args.question:
        print("empty query — give something to search for", file=sys.stderr)
        return 2
    loaded = _load_model(conn, directory, args.model)
    if loaded is None:
        return 2
    embedder, model_id, store = loaded
    from dyprys.search import flat_search, resolve, scanned_fraction, scope_books

    books = None
    if args.collection:
        books = scope_books(conn, args.collection)
        if not books:
            print(f"no book matches {args.collection!r}; try `dyp books`", file=sys.stderr)
            return 1

    router, problem = _router(conn, directory, embedder, model_id, args.route, books)
    if problem:
        return 2
    reranker, missing = _load_reranker(args, conn)
    if missing:
        return 2
    expander, missing = _load_expander(args, conn)
    if missing:
        print(missing, file=sys.stderr)
        return 2

    stages = Stages(not args.quiet)
    started = time.time()
    search = _searcher(conn, store, embedder, model_id, args.mode, books, router,
                       reranker, args.rerank or args.depth or 0, args.dedupe,
                       expander, stages)
    hits = search(args.question, args.k)
    elapsed = (time.time() - started) * 1000
    reached = router.last.get("books") if router else books
    share = scanned_fraction(conn, model_id, reached)

    # The machine-readable form. An agent driving dyp parses this instead of
    # scraping the human display, which carries ANSI codes and middle-elided
    # titles. Empty results are valid JSON, not a stderr message, so a caller
    # gets one shape to parse either way; the exit code still says hit or miss.
    if getattr(args, "json", False):
        import json as _json
        payload = results_as_json(
            args.question, args.mode, bool(args.route), share, elapsed,
            resolve(conn, hits), search.why, search.cosine)
        print(_json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["results"] else 1

    if not hits:
        embedded = conn.execute(
            "SELECT COALESCE(SUM(n_embedded), 0) FROM segment_progress WHERE model_id = ?",
            (model_id,)).fetchone()[0]
        if not embedded:
            print("nothing embedded yet — run `dyp embed` first", file=sys.stderr)
        else:
            # A genuinely empty result on an embedded index: vector search always
            # returns something, so this is a lexical-only mode finding no words.
            print("no passage matched", file=sys.stderr)
        return 1
    passages = resolve(conn, hits)
    for rank, p in enumerate(passages, 1):
        where = term.bold(p.title) + (
            term.dim(f", chapter {p.chapter + 1}") if p.chapter else "")
        # Not the fused score. That is sum(1/(60+rank)), so it spans 0.016 to
        # 0.033 for two lists — a perfect match prints as 0.033, which reads as
        # 3%, and two passages that each led one half tie exactly. The rank each
        # half gave it is the real basis for the order, and it says whether the
        # passage was found because it *means* the query or *contains* it.
        why = search.why.get(p.chunk_id) or "—"
        cos = search.cosine.get(p.chunk_id)
        strength = f"cos {cos:.2f} · " if cos is not None else ""
        print(f"\n{term.bold(str(rank) + '.')} {term.dim('[' + strength + why + ']')} "
              f"{where}  {term.dim('(chunk ' + str(p.chunk_id) + ')')}")
        if p.text is None:
            print("   " + _WHY_NO_TEXT.get(p.state, "(source unavailable; run `dyp check`)"))
        else:
            if p.state == text_mod.SHIFTED:
                print("   (the file was edited above this passage; found intact"
                      " further on — re-run `dyp add` to update the offsets)")
            # Read with a little either side. A chunk boundary is chosen by byte
            # budget, so it lands mid-sentence often; the extract used to begin
            # wherever the chunk did, which reads as though the text were
            # damaged. `read_window` starts and ends on a sentence instead.
            widened = text_mod.read_window(p.path, p.offset, len(p.text.encode()))
            body, at = widened if widened else (p.text, p.offset)
            flat = " ".join(body.split())
            wanted = term.terms(args.question)
            if args.full:
                shown, cut_before, cut_after = flat, False, False
            else:
                # Which part of a 3,500-byte passage to show is a real choice,
                # and the opening is merely where the chunker happened to cut.
                shown, cut_before, cut_after = term.best_extract(
                    flat, wanted, EXTRACT_CHARS)
            shown = ("… " if cut_before else "") + shown + (" …" if cut_after else "")
            # Wrapped before marking: escape codes have no width, and wrapping
            # after would count them and produce ragged lines.
            wrapped = textwrap.fill(shown, width=94, initial_indent="   ",
                                    subsequent_indent="   ")
            print(term.mark(wrapped, wanted))
            print(term.dim(f"   {_where_to_look(p.path, at)}"))
    if getattr(args, "summarise", None):
        _summarise(args, passages, conn, stages, search)

    if args.route:
        # "of the vectors", not "of the corpus": an exact phrase is still looked
        # up across the whole lexical index, which is what keeps a remembered
        # sentence findable when stage 1 would have routed past its book.
        scope = f"{len(reached)} routed book(s), {share:.1%} of the vectors"
    elif books:
        scope = f"{len(books)} book(s), {share:.1%} of the vectors"
    else:
        scope = "the whole embedded corpus"
    print(f"\n{elapsed:.1f} ms, {args.mode} over {scope}", file=sys.stderr)

    # Kept for the person, not for the search. A question worth asking twice
    # should not have to be reconstructed from memory, and a good answer should
    # be findable again — `dyp asked`.
    db.record_question(conn, args.question, args.mode, elapsed, {
        "routed": bool(router),
        "books": len(reached) if reached else None,
        "models": {k: v for k, v in (
            ("embed", _shorten(embedder.name)),
            ("expander", getattr(expander, "model", None) if expander else None),
            ("summariser", _last_summariser.get("model")),
        ) if v},
        "expansion": _last_expansion.get("lines") or None,
        "hits": [{"chunk": p.chunk_id, "title": p.title, "path": p.path,
                  "offset": p.offset, "cos": round(search.cosine.get(p.chunk_id, 0.0), 3),
                  "why": search.why.get(p.chunk_id)} for p in passages],
        "answer": _last_summariser.get("answer"),
    })
    return 0


def _lexical(conn) -> int:
    """Backfill the BM25 index. New ingests populate it inline and skip this."""
    from dyprys.lexical import backfill

    started = time.time()

    def show(done, total):
        print(f"\r  {done:,}/{total:,} chunks indexed ", end="", file=sys.stderr, flush=True)

    n = backfill(conn, progress=show)
    print(file=sys.stderr)
    db.record_event(conn, "lexical", f"rebuilt the BM25 index over {n:,} chunks")
    print(f"indexed {n:,} chunks in {time.time() - started:.1f}s")
    return 0


def _remove(conn, args) -> int:
    """Drop books from the index. Reclaiming their ids is a separate step."""
    from dyprys.compact import remove_books
    from dyprys.library import books as inspect

    found = inspect(conn, args.pattern)
    if not found:
        print(f"no book matches {args.pattern!r}", file=sys.stderr)
        return 1

    chunks = sum(b.chunks for b in found)
    print(f"this would remove {len(found)} book(s) and {chunks:,} chunks:")
    for book in found[:10]:
        print(f"  {book.title}")
    if len(found) > 10:
        print(f"  … and {len(found) - 10} more")
    print("\ntheir chunk ids become dead space until `dyp compact` reclaims them.")
    print("the source files on disk are not touched.")
    if not args.yes:
        print("\nre-run with --yes to go ahead.", file=sys.stderr)
        return 1

    remove_books(conn, {b.id for b in found})
    db.record_event(conn, "remove",
                    f"{len(found)} books matching {args.pattern!r}, {chunks:,} chunks "
                    f"left reclaimable")
    print(f"\nremoved {len(found)} book(s); {chunks:,} chunks are now reclaimable")
    return 0


def _compact(conn, directory, args) -> int:
    """Close the gaps in the chunk id space, rewriting every store."""
    from dyprys.compact import compact, interrupted, outstanding_carries, plan
    from dyprys.lock import AlreadyRunning, exclusive

    if interrupted(conn):
        print("a previous compaction stopped part-way through; resuming it")
    else:
        waiting = outstanding_carries(conn)
        if waiting:
            print(f"{waiting:,} vectors are still waiting to be carried across an edit.",
                  file=sys.stderr)
            print("run `dyp embed` first — compacting now would move the rows they "
                  "are waiting to copy.", file=sys.stderr)
            return 1
        shape = plan(conn)
        if not shape.worth_doing:
            print(f"nothing to reclaim — all {shape.live:,} chunk ids are live")
            return 0
        print(f"this would reclaim {shape.dead:,} dead chunk ids, leaving {shape.live:,}.")
        print("every vector file, chunk row, segment start and the BM25 index is rewritten.")
        if not args.yes:
            print("\nre-run with --yes to go ahead.", file=sys.stderr)
            return 1

    started = time.time()

    def show(model_id, done, total):
        print(f"\r  model {model_id}: {done}/{total} runs moved   ",
              end="", file=sys.stderr, flush=True)

    try:
        # Shares the embed lock: both rewrite the vector files, and running them
        # together would move rows out from under an embedding pass.
        with exclusive(directory, "embed"):
            report = compact(conn, directory, progress=show)
    except AlreadyRunning as busy:
        print(f"\n{busy}", file=sys.stderr)
        return 2
    except RuntimeError as blocked:
        print(f"\n{blocked}", file=sys.stderr)
        return 1

    print(file=sys.stderr)
    if report.resumed:
        print("resumed and finished a compaction that had been interrupted")
    db.record_event(conn, "compact",
                    f"reclaimed {report.reclaimed:,} chunk ids, {report.live:,} live "
                    f"across {report.models} model(s)")
    print(f"reclaimed  {report.reclaimed:>10,} chunk ids in {time.time() - started:.1f}s")
    print(f"live       {report.live:>10,} chunks across {report.models} model(s)")
    return 0


def _route(conn, directory, args) -> int:
    """Build stage 1. Re-runnable; replaces whatever was there."""
    loaded = _load_model(conn, directory, args.model)
    if loaded is None:
        return 2
    embedder, model_id, store = loaded
    from dyprys.routing import build_centroids

    started = time.time()

    def show(done, total, title):
        name = title if len(title) <= 40 else title[:39] + "…"
        print(f"\r  {done}/{total} books profiled  {name:<41}",
              end="", file=sys.stderr, flush=True)

    report = build_centroids(conn, directory, store, model_id,
                             per_book=args.centroids, progress=show)
    print(file=sys.stderr)
    if not report.books:
        print(f"nothing to profile: none of the {report.skipped:,} books has been "
              f"embedded yet.", file=sys.stderr)
        print("run `dyp embed` first — routing summarises vectors, so it needs them "
              "to exist.", file=sys.stderr)
        return 1
    db.record_event(conn, "route",
                    f"profiled {report.books:,} books into {report.centroids:,} directions")
    print(f"profiled   {report.books:>6,} books into {report.centroids:,} directions "
          f"in {time.time() - started:.1f}s")
    if report.skipped:
        print(f"skipped    {report.skipped:>6,} books with nothing embedded yet")
    return 0


def _load_reranker(args, conn=None):
    """The cross-encoder, or None when reranking was not asked for."""
    if not args.rerank:
        return None, None
    path = resolve_model(conn, "reranker", "DYPRYS_RERANKER", args.reranker)
    if not path or not Path(path).exists():
        print(_no_model_for("reranker", "--reranker", "DYPRYS_RERANKER", chat=False),
              file=sys.stderr)
        return None, "missing"
    from dyprys.rerank import Reranker

    print(f"loading {Path(path).name} …", file=sys.stderr)
    return Reranker(path), None


# Whether a rewritten query is also given to BM25. It is not, and the reason is
# the same one that already sends an exact phrase past the router: a phrase is a
# lookup, not a ranking, and a lookup cannot be improved by rewording the thing
# being looked up. Measured -- see the docstring of `dyprys.expand`.
EXPAND_LEXICAL = False


# How much of a passage to print. A chunk averages 3,534 bytes; showing 400 of
# them was 11% and always ended in an ellipsis, so nothing said whether the
# passage stopped there or the display did. `--full` prints the whole thing.
EXTRACT_CHARS = 700


def _stderr_is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


# Filled by the stages that produce them, read once when the question is
# recorded. Module-level rather than threaded through five signatures, because
# every one of those signatures belongs to something that should not have to
# know a history exists.
_last_expansion: dict = {}
_last_summariser: dict = {}
_retried: dict = {}


class Stages:
    """Says what each stage did, on stderr, and leaves it on the screen.

    An expanded and summarised query takes ten seconds or more, and silence for
    ten seconds is indistinguishable from a hang. An earlier version wrote a
    transient line and erased it, which meant the interesting part -- the
    rephrasing the model chose, the books routing picked -- flashed past and was
    gone. Nothing is erased now: every line is something worth having kept.

    stderr, so `dyp ask ... > results.txt` still captures only results.
    """

    def __init__(self, on: bool = True):
        self.on = bool(on)

    def note(self, label: str, detail: str = "", *, indent: int = 0) -> None:
        if not self.on:
            return
        pad = "  " + "    " * indent
        if detail:
            print(f"{pad}{term.dim(label + ':')} {detail}", file=sys.stderr)
        else:
            print(f"{pad}{term.dim(label)}", file=sys.stderr)

    def working(self, message: str) -> None:
        """Announced before a slow stage, so the wait has a name."""
        self.note(message)


def _where_to_look(path: str, offset: int) -> str:
    """The file and byte a reader can open to see the passage in place.

    `path:offset` because that is what editors take: `less +1054692P file`,
    or a jump-to-byte in anything else. A chunk id is ours and means nothing
    outside this index.
    """
    home = str(Path.home())
    shown = path.replace(home, "~", 1) if path.startswith(home) else path
    return f"{shown}:{offset}"


def _widen(passages):
    """Passages re-read with context, and where each chunk sits inside its window.

    A chunk boundary is a byte budget, so a passage handed to a model often
    begins mid-sentence and it copies the fragment. The span is kept so a quote
    can be told apart from one taken out of the margin either side.
    """
    import dataclasses

    widened, spans = [], {}
    for p in passages:
        if p.text is None:
            widened.append(p)
            continue
        length = len(p.text.encode())
        window = text_mod.read_window(p.path, p.offset, length)
        if not window:
            widened.append(p)
            continue
        text, at = window
        widened.append(dataclasses.replace(p, text=text, offset=at))
        head = len(text.encode()[: p.offset - at].decode("utf-8", errors="ignore"))
        spans[p.chunk_id] = (head, head + len(text[head:].encode()[:length]
                                              .decode("utf-8", errors="ignore")))
    return widened, spans


def _summarise(args, passages, conn=None, stages=None, search=None) -> None:
    """Draft an answer from what was found, and show only what checks out.

    Printed after the passages, never instead of them. The passages are the
    result; this is a reading of them, and a reading that cannot be verified is
    worth less than the list it was drawn from.
    """
    from dyprys import summarise as summarise_mod
    from dyprys.summarise import NO_ANSWER, ask_ollama, summarise

    model = resolve_model(conn, "summariser", "DYPRYS_SUMMARISER",
                          args.summarise if isinstance(args.summarise, str) else None)
    if not model:
        print("\n" + _no_model_for("summariser", "--summarise", "DYPRYS_SUMMARISER"),
              file=sys.stderr)
        return

    # The model reads the same widened passages the reader sees, so a chunk cut
    # mid-sentence does not become a fragment it has to guess around. Verifying
    # against the widened text is still exact: it is the text that was shown,
    # read from the file at a recorded byte.
    import dataclasses

    widened, spans = _widen(passages)

    talk = lambda p: ask_ollama(model, p)
    answer = summarise(args.question, widened, talk)

    # A refusal means the *search* failed, not that the library lacks an answer,
    # and a failed search can be tried with other words. This costs nothing when
    # the first attempt works, and only ever runs when the alternative is
    # nothing at all. One retry: a model that cannot find it twice is telling
    # you something, and a loop here would spend minutes proving it.
    if answer.prose.strip() == NO_ANSWER and search is not None:
        if stages:
            stages.note("nothing here answers it — asking the same model for "
                        "other words")
        other = summarise_mod.rephrase(args.question, talk)
        if other:
            if stages:
                stages.note("trying instead", other, indent=1)
            from dyprys.search import resolve as _resolve
            again = _resolve(conn, search(other, args.k))
            widened, spans = _widen(again)
            second = summarise(args.question, widened, talk)
            if second.prose.strip() != NO_ANSWER:
                answer, passages = second, again
                _retried["question"] = other
            else:
                _retried["failed"] = other
                # The retry's passages are discarded, so the widened set must go
                # back to the ones on screen: a citation has to name something
                # the reader can see.
                widened, spans = _widen(passages)
    _last_summariser.update(model=model, retried=_retried.get("question"), answer={
        "prose": answer.prose.strip(),
        "verified": [{"cited": c.cited, "quote": c.quote, "where": c.location}
                     for c in answer.verified],
        "rejected": [{"cited": c.cited, "quote": c.quote} for c in answer.rejected]})
    print("\n" + term.rule())
    if _retried.get("question"):
        print(term.dim(f"the passages above did not answer, so the library was "
                       f"searched again for:"))
        print(f"  {term.bold(_retried['question'])}\n")
        for rank, p in enumerate(passages, 1):
            print(f"{term.bold(str(rank) + '.')} {term.bold(p.title)}"
                  f"  {term.dim('(chunk ' + str(p.chunk_id) + ')')}")
            print(term.dim(f"   {_where_to_look(p.path, p.offset)}"))
        print()
    if not answer.prose.strip():
        print(f"{model} returned nothing; the passages above are unaffected.",
              file=sys.stderr)
        return
    if answer.prose.strip() == NO_ANSWER:
        print("the model reports that these passages do not answer the question.")
        if _retried.get("failed"):
            print(term.dim(f"  it was asked again for “{_retried['failed']}”, "
                           f"and found nothing there either."))
        return

    print(answer.prose.strip())
    print(term.dim(f"\n— drafted by {model} —"))
    if answer.verified:
        print(term.bold("\nchecked against the source:"))
        for claim in answer.verified:
            kind = "quoted" if claim.quoted else "verbatim, unquoted"
            shown = widened[claim.cited - 1].text or ""
            at = shown.find(claim.quote)
            span = spans.get(claim.chunk_id)
            if at >= 0 and span and not (span[0] <= at < span[1]):
                kind += ", from the context either side"
            print(f"  {term.style('✓', 'green')} [{claim.cited}] {kind} — {claim.location}")
    if answer.rejected:
        # The interesting output. A quote that is not in the passage it cites is
        # the failure this exists to catch, and hiding it would waste the catch.
        quoted = sum(1 for c in answer.rejected if c.quoted)
        print(f"\n{len(answer.rejected)} claim(s) not found in the passage cited"
              + (f", {quoted} of them quoted:" if quoted else ":"))
        for claim in answer.rejected:
            mark = "misquotation" if claim.quoted else "paraphrase"
            print(f"  ✗ [{claim.cited}] {mark}: {claim.quote[:64]}")
    if not answer.claims:
        print("\n(no quotations offered, so nothing here has been checked)")


def _no_model_for(role: str, flag: str, env: str, chat: bool = True) -> str:
    """A refusal that names something the user can actually run.

    Naming the flag is not help. Someone who has never installed ollama and owns
    no .gguf learns nothing from being told the flag exists, and every optional
    role used to fail that way in three different wordings.

    `chat=False` for reranking, which needs a **cross-encoder** and not a chat
    model: it scores a (query, passage) pair rather than generating, so an
    ollama chat name is not a candidate for it however many are installed.
    Offering one would be a suggestion that cannot work.
    """
    lines = [f"no {role} model. Pass one: `{flag} MODEL`, or set ${env}."]
    if not chat:
        lines.append("  this one needs a cross-encoder .gguf, not a chat model —")
        lines.append("  e.g. qwen3-reranker-0.6b. An ollama name will not do.")
        lines.append("  It reliably gets the answer into the top 5 (+28/-2 over")
        lines.append("  four arms) and rarely to rank 1. ~0.9s per passage, so")
        lines.append("  ~9s a query: worth it if you read all 5, not to type at.")
        return "\n".join(lines)
    installed = ollama_models()
    if installed:
        lines.append(f"  ready to use now:  {'   '.join(installed[:6])}")
        lines.append(f"  e.g.  dyp ask \"your question\" {flag} gemma3:4b")
    else:
        lines.append("  none installed: `ollama serve` then `ollama pull gemma3:4b`,")
        lines.append("  or pass a path to any instruct .gguf.")
    lines.append("  `dyp models` lists every optional role and what is available.")
    return "\n".join(lines)


def _load_expander(args, conn=None):
    """The expansion model, or None when expansion was not asked for.

    Returns `(expander, missing)` like `_load_reranker`, so a missing model is
    reported by the caller rather than raising out of the middle of a search.
    """
    asked = getattr(args, "expand", None)
    if not asked:
        return None, None
    # `--expand gemma3:4b` names the model inline, as `--summarise` does;
    # `--expander` stays as the older spelling.
    named = resolve_model(
        conn, "expander", "DYPRYS_EXPANDER",
        (asked if isinstance(asked, str) else None) or getattr(args, "expander", None))
    if not named:
        return None, _no_model_for("expansion", "--expand", "DYPRYS_EXPANDER")
    # A path to weights we load ourselves; anything else is a name for the local
    # ollama server, which owns each model's chat template. Guessing a template
    # is how the reranker was made worse than no reranker at all.
    if named.endswith(".gguf") or Path(named).exists():
        from dyprys.expand import Expander

        return Expander(named), None
    from dyprys.expand import OllamaExpander

    expander = OllamaExpander(named, prompt=os.environ.get("DYPRYS_PROMPT", "full"))
    problem = expander.unavailable()
    return (None, problem) if problem else (expander, None)


def _searcher(conn, store, embedder, model_id, mode, books, router=None,
              reranker=None, depth=0, dedupe=False, expander=None, stages=None):
    """One callable per mode, so ask and eval cannot drift apart.

    `router` narrows the books per query, which is stage 1. It applies to the
    vector half and to BM25's OR-of-words ranking, but *not* to BM25's exact
    phrase attempt, which searches the library whole -- see `search_bm25`. So
    the reported scanned fraction is the share of the *vectors* read, and the
    output says so.
    """
    from dyprys.lexical import reciprocal_rank_fusion, search_bm25
    from dyprys.search import flat_search

    def run(question, k):
        # Expansion first, because everything below may be run several times
        # over. Routing still sees the question as asked: stage 1 picking books
        # from a rewritten query is a separate change with its own measurement,
        # and mixing the two would leave neither attributable.
        note = stages.note if stages else (lambda *a, **k: None)
        working = stages.working if stages else (lambda _m: None)
        expansion = None
        if expander:
            working(f"rephrasing with {getattr(expander, 'model', 'a model')} …")
            expansion = expander.expand(question)
            _last_expansion["lines"] = {
                "vector": list(expansion.vector), "hyde": list(expansion.hyde),
                "lexical": list(expansion.lexical)}
            for kind, lines in (("as a question", expansion.vector),
                                ("as an answer", expansion.hyde),
                                ("as keywords", expansion.lexical)):
                for line in lines:
                    note(kind, line[:120], indent=1)
            if expansion.empty:
                note("nothing usable came back; searching as asked", indent=1)
        vector = embedder.embed_query(question)
        scope = router(question, vector) if router else books
        if router and scope:
            total = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
            titles = [r["title"] for r in conn.execute(
                "SELECT title FROM books WHERE id IN (%s)" % ",".join("?" * len(scope)),
                sorted(scope))]
            note(f"routed to {len(scope)} of {total:,} books")
            for title in titles:
                note(title[:76], indent=1)
        # Retrieval fetches a deeper shortlist when something downstream will
        # thin or reorder it; recall@depth of this list is the ceiling on what
        # reranking can reach.
        #
        # Depth is deliberately independent of the reranker. Tying the two
        # together made an arm change two things at once -- a reranked run
        # retrieved deeper *and* reordered -- and the two are separable: jina's
        # recall@1 is 63/110 at depth 5 and 67/110 at depth 20 with no reranker
        # at all, because RRF fuses however many candidates each half returned.
        # Crediting that +4 to the reranker overstated it, and did so in the
        # direction the measurement was hoping for.
        want = max(k, depth)
        # Over-fetch when a filter will thin the list, or it cannot return k.
        if dedupe:
            want = max(want, k * 4)
        # The query as asked leads each half, and leads the fused list, because
        # RRF breaks ties in favour of the earlier ranking. A rewrite is a guess
        # about what was meant; it may add an answer the original missed, but it
        # does not get to overrule the original on a tie.
        by_vector, by_words = [], []
        origin: dict[int, str] = {}
        if mode in ("vector", "hybrid"):
            by_vector.append(flat_search(conn, store, vector, model_id, want, book_ids=scope))
        if mode in ("lexical", "hybrid"):
            by_words.append(search_bm25(conn, question, want, model_id=model_id,
                                        book_ids=scope, origin=origin))

        if expansion and not expansion.empty:
            if mode in ("vector", "hybrid"):
                # A rewritten question is a query; a hypothetical answer is a
                # passage. EmbeddingGemma is trained with a different prefix for
                # each, and the model's own line kinds say which is which.
                for phrase in expansion.vector:
                    by_vector.append(flat_search(
                        conn, store, embedder.embed_query(phrase), model_id, want, book_ids=scope))
                for passage in expansion.hyde:
                    by_vector.append(flat_search(
                        conn, store, embedder.embed_documents([passage])[0],
                        model_id, want, book_ids=scope))
            if mode in ("lexical", "hybrid") and EXPAND_LEXICAL:
                for phrase in expansion.lexical:
                    by_words.append(search_bm25(
                        conn, phrase, want, model_id=model_id, book_ids=scope))

        # Each half is fused down to ONE ranking before the halves meet, so the
        # balance between them does not depend on how many phrasings the model
        # happened to emit for each. Handing fusion four vector lists and two
        # lexical ones is a 2:1 vector weight arrived at by accident, and a
        # deliberate 1.3x weight was already measured to cost 24 of 30
        # exact-phrase probes. The first version did exactly that, and lexical
        # safety fell 18/20 -> 13/20 on two separate question sets.
        def fused(rankings):
            rankings = [r for r in rankings if r]
            if not rankings:
                return []
            if len(rankings) == 1:
                return rankings[0]
            return reciprocal_rank_fusion(rankings, k=want)

        vector_side, lexical_side = fused(by_vector), fused(by_words)
        sides = [r for r in (vector_side, lexical_side) if r]
        shortlist = sides[0] if len(sides) == 1 else reciprocal_rank_fusion(sides, k=want)

        if reranker:
            working(f"rescoring {min(len(shortlist), want)} passages …")
            from dyprys.rerank import rerank
            shortlist = rerank(conn, reranker, question, shortlist[:want], max(k * 4, k))
        if dedupe:
            from dyprys.search import drop_near_duplicates
            shortlist = drop_near_duplicates(conn, shortlist, k)
        found = shortlist[:k]
        # Attached rather than returned, so `eval` and every other caller keep
        # the same signature. Same pattern as `router.last`.
        from dyprys.lexical import provenance
        # "phrase" and "words" rather than one "bm25": an exact-phrase hit and a
        # keyword-overlap hit are different kinds of answer, and the reader is
        # the one who knows which they wanted.
        run.why = provenance(
            [n for n in (("vec", vector_side), ("bm25", lexical_side)) if n[1]],
            found, rename={"bm25": origin})
        # Cosine for *every* result, including the ones only BM25 found — k dot
        # products against vectors already on disk. A literal match with a low
        # cosine is worth seeing: it says the passage contains your words without
        # being about them, which is exactly when a keyword hit misleads.
        run.cosine = {}
        for hit in found:
            try:
                run.cosine[hit.chunk_id] = float(store.read(hit.chunk_id) @ vector)
            except (IndexError, ValueError):
                pass
        return found

    run.why, run.cosine = {}, {}
    return run


def _router(conn, directory, embedder, model_id, books_wanted, candidates):
    """Stage 1 as a callable, or None when routing was not asked for."""
    if not books_wanted:
        return None, None
    from dyprys.routing import is_built, route
    from dyprys.vectors import VectorStore

    if not is_built(conn, model_id):
        print("no routing profile yet — run `dyp route`", file=sys.stderr)
        return None, "missing"

    total = conn.execute(
        "SELECT COALESCE(SUM(centroid_count), 0) FROM book_centroids WHERE model_id = ?",
        (model_id,),
    ).fetchone()[0]
    centroids = VectorStore(db.centroids_path(directory, model_id), embedder.dim, total)
    last: dict = {}

    def go(question, vector):
        chosen = {b for b, _ in route(conn, centroids, vector, model_id, books_wanted, candidates)}
        last["books"] = chosen
        return chosen

    go.last = last
    return go, None


def _size(n: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def _books(conn, pattern=None, as_json: bool = False) -> int:
    """The library, or one book in full."""
    from dyprys.library import books as inspect

    found = inspect(conn, pattern)
    if not found:
        if as_json:
            return _emit_json({"books": []}) or 1
        where = f" matching {pattern!r}" if pattern else ""
        print(f"no books{where} — run `dyp add`" if not pattern else f"no book matches {pattern!r}",
              file=sys.stderr)
        return 1

    if as_json:
        return _emit_json({"books": [
            {
                "title": b.title,
                "key": str(b.key),
                "chunks": b.chunks,
                "lexical_indexed": b.lexical,
                "sources": [
                    {"ordinal": s.ordinal, "path": str(s.path),
                     "size_bytes": s.size_bytes, "present": s.present}
                    for s in b.sources],
                "chunkings": [
                    {"id": c.id, "target": c.target, "overlap": c.overlap,
                     "chunks": c.chunks} for c in b.chunkings],
                "embedded": dict(b.per_model),
            }
            for b in found]})

    if pattern:
        for book in found:
            print(f"\n{book.title}")
            print(f"  key            {book.key}")
            missing = [s for s in book.sources if not s.present]
            print(f"  sources        {len(book.sources)}"
                  f"{f' ({len(missing)} missing from disk)' if missing else ''}")
            for src in book.sources[:12]:
                mark = " " if src.present else "!"
                print(f"   {mark} {src.ordinal:>3}. {_size(src.size_bytes):>8}  {src.path}")
            if len(book.sources) > 12:
                print(f"       … and {len(book.sources) - 12} more")
            for ch in book.chunkings:
                print(f"  chunking {ch.id}     {ch.target}B / {ch.overlap}B overlap"
                      f"  →  {ch.chunks:,} chunks")
            print(f"  BM25 indexed   {book.lexical:,} of {book.chunks:,}")
            for name, done in book.per_model.items():
                share = f"{done / book.chunks:.0%}" if book.chunks else "--"
                print(f"  {_shorten(name, 14):<14} {done:>7,} embedded  {share}")
        return 0

    width = min(52, max(len(b.title) for b in found))
    models = sorted({name for b in found for name in b.per_model})
    header = "".join(f"  {_shorten(n, 12):>12}" for n in models)
    print(f"{'title':<{width}}  {'chunks':>8}  {'files':>5}  {'bm25':>6}{header}")
    for book in found:
        title = book.title if len(book.title) <= width else "…" + book.title[-(width - 1):]
        lex = "all" if book.lexical == book.chunks else f"{book.lexical:,}"
        per = ""
        for name in models:
            done = book.per_model.get(name, 0)
            if not book.chunks:
                per += f"  {'—':>12}"
            elif done >= book.chunks:
                per += f"  {'all':>12}"
            elif done:
                per += f"  {done / book.chunks:>11.0%} "
            else:
                per += f"  {'':>12}"
        print(f"{title:<{width}}  {book.chunks:>8,}  {len(book.sources):>5}  {lex:>6}{per}")

    chunks = sum(b.chunks for b in found)
    print(f"\n{len(found):,} books, {chunks:,} chunks")
    # With thousands of books the per-row column is a haystack; the distribution
    # is the thing you actually came to see.
    for name in models:
        done = sum(min(b.per_model.get(name, 0), b.chunks) for b in found)
        complete = sum(1 for b in found if b.chunks and b.per_model.get(name, 0) >= b.chunks)
        started = sum(1 for b in found
                      if 0 < b.per_model.get(name, 0) < b.chunks)
        untouched = len(found) - complete - started
        print(f"\n{_shorten(name)}")
        print(f"  {done:>10,} of {chunks:,} chunks  {done / chunks:.0%}" if chunks else "")
        print(f"  {complete:>10,} books complete")
        if started:
            print(f"  {started:>10,} book(s) part-way")
        print(f"  {untouched:>10,} books not started")
    print("\nrun `dyp books PATTERN` for detail on one")
    return 0


def _models_needed(conn) -> int:
    """What weights this index was built with, in enough detail to go and get them."""
    rows = db.model_provenance(conn)
    if not rows:
        print("this index has no models, so nothing is needed", file=sys.stderr)
        return 1
    for row in rows:
        print(f"{row['name']}")
        print(f"  dimensions   {row['dim']}")
        if row["file_name"]:
            print(f"  file         {row['file_name']}")
        if row["file_bytes"]:
            print(f"  size         {_size(row['file_bytes'])}")
        if row["file_sha256"]:
            print(f"  sha256       {row['file_sha256']}")
        print(f"  source       {row['source_uri'] or 'not recorded — `dyp models --source` sets it'}")
        if not row["file_sha256"]:
            print("  (this index predates provenance recording; embedding or querying "
                  "once with the right weights fills it in)")
    print("\nverify a candidate before using it:  dyp models --verify PATH.gguf")
    return 0


def _models_verify(conn, candidate: Path) -> int:
    """Is this file the one the index's vectors were made with?

    Worth answering before rather than after: the wrong file does not fail, it
    quietly embeds the whole library again under a new identity.
    """
    if not candidate.exists():
        print(f"no such file: {candidate}", file=sys.stderr)
        return 2
    rows = db.model_provenance(conn)
    if not rows:
        print("this index has no models to check against", file=sys.stderr)
        return 1

    from dyprys.embedder import _file_digest

    size = candidate.stat().st_size
    print(f"checking {candidate.name}  ({_size(size)})", file=sys.stderr)
    digest = _file_digest(candidate)

    for row in rows:
        if row["file_sha256"] == digest:
            print(f"matches {row['name']}")
            print("  its vectors apply; nothing needs re-embedding")
            return 0
    print(f"sha256 {digest[:16]}… matches none of this index's models:")
    for row in rows:
        known = (row["file_sha256"] or "")[:16]
        print(f"  {row['name']}  expects {known + '…' if known else '(not recorded)'}")
    print("\nembedding with this file would start from zero under a new identity.",
          file=sys.stderr)
    return 1


# The three optional roles. None of them is stored in the index, none embeds
# anything, and all three are off by default -- which is exactly why they are
# invisible until something fails. `dyp models` lists the embedding model
# because the index has one; these have to be listed because it does not.
OPTIONAL_ROLES = (
    ("expansion", "DYPRYS_EXPANDER", "--expand MODEL", "expander",
     "rephrases the query before searching  (+9 recall@1, ~11x slower)"),
    ("summarising", "DYPRYS_SUMMARISER", "--summarise MODEL", "summariser",
     "drafts an answer, quoting only what verifies against the source"),
    ("reranking", "DYPRYS_RERANKER", "--rerank --reranker PATH.gguf", "reranker",
     "reorders the shortlist  (+28/-2 into the top 5; ~9s a query)"),
)


def default_model(conn, role: str) -> str | None:
    """The model remembered for `role` in this index, if any."""
    return db.get_meta(conn, f"model.{role}") if conn is not None else None


def resolve_model(conn, role: str, env: str, explicit=None) -> str | None:
    """Which model to use: what was typed, else the environment, else the index.

    Explicit beats remembered, exactly as `--data` beats the default library.
    A command that named its model must never be redirected by a setting made
    weeks ago, and the environment sits between the two because it is scoped to
    the shell you are in.
    """
    if isinstance(explicit, str) and explicit:
        return explicit
    return os.environ.get(env) or default_model(conn, role)


def ollama_models(host: str = "http://localhost:11434", timeout: float = 0.7) -> list[str]:
    """What the local ollama server has, or [] if it is not running.

    A short timeout on purpose. This runs inside `dyp models`, which most users
    will run without ollama installed, and a listing command that pauses for
    three seconds to discover nothing is worse than one that says nothing. The
    server is local: it answers in milliseconds or it is not there.
    """
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            return sorted(m["name"] for m in json.loads(response.read()).get("models", []))
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return []


def _print_optional_models(conn=None) -> None:
    """Say that the optional roles exist, and name something usable for them."""
    print("\noptional models — never used to embed, all off by default")
    for role, env, flag, key, what in OPTIONAL_ROLES:
        remembered, from_env = default_model(conn, key), os.environ.get(env)
        if from_env:
            state = f"{from_env}  (${env})"
        elif remembered:
            state = f"{remembered}  (default for this library)"
        else:
            state = "not set"
        print(f"  {role:<13} {flag:<28} {state}")
        print(f"  {'':<13} {what}")
    print(f"\n  set a default:  dyp models --{OPTIONAL_ROLES[0][3]} gemma3:4b")
    installed = ollama_models()
    if installed:
        print(f"\n  ready to use now:  {'   '.join(installed[:6])}")
        print("  gemma3:4b is the one measured best for expansion and summarising.")
    else:
        print("\n  none available: start `ollama serve` and `ollama pull gemma3:4b`,")
        print("  or pass a path to any instruct .gguf.")


def _env_for(key: str) -> str:
    """The environment variable that overrides this role's stored default."""
    return next(env for _r, env, _f, k, _w in OPTIONAL_ROLES if k == key)


def _set_default_model(conn, key: str, value: str) -> int:
    """Remember a default optional model, after checking it can be used.

    Validated when it is set rather than when it is next needed, because the
    two can be weeks apart: a typo stored today should not surface as a failed
    search in a fortnight.
    """
    if value.lower() in ("none", "off", ""):
        with conn:
            conn.execute("DELETE FROM meta WHERE key = ?", (f"model.{key}",))
        print(f"{key}: no default; --{key} or ${_env_for(key)} still work per query")
        return 0

    if not (value.endswith(".gguf") or Path(value).exists()):
        installed = ollama_models()
        if installed and value not in installed and f"{value}:latest" not in installed:
            print(f"ollama has no model named {value!r}.", file=sys.stderr)
            print(f"  installed:  {'   '.join(installed[:6])}", file=sys.stderr)
            print(f"  or:  ollama pull {value}", file=sys.stderr)
            return 1
    elif not Path(value).exists():
        print(f"no such file: {value}", file=sys.stderr)
        return 1

    with conn:
        db.set_meta(conn, f"model.{key}", value)
    db.record_event(conn, "default", f"{key} = {value}")
    print(f"{key}: {value}  — used by this library unless a flag or "
          f"${_env_for(key)} says otherwise")
    return 0


def _models(conn, directory, args) -> int:
    """What has been embedded, by what, at what cost on disk."""
    from dyprys.library import drop_model, models as inspect

    for _role, _env, _flag, key, _what in OPTIONAL_ROLES:
        chosen = getattr(args, key, None)
        if chosen:
            return _set_default_model(conn, key, chosen)

    found = inspect(conn, directory)
    if not found:
        if getattr(args, "json", False):
            return _emit_json({"models": []}) or 1
        print("no embedding model registered yet — run `dyp embed`", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        return _emit_json({"models": [
            {
                "name": m.name,
                "alias": m.alias,
                "dim": m.dim,
                "store": m.quantisation,
                "embedded": m.embedded,
                "live_chunks": m.live_chunks,
                "coverage": m.coverage,
                "disk_bytes": m.bytes_on_disk,
                "failures": m.failures,
                "carries": m.carries,
                "routing": {"profiled_books": m.centroid_books,
                            "stale_books": m.stale_books},
                "file_path": m.file_path,
                "file_present": bool(m.file_path) and Path(m.file_path).exists(),
            }
            for m in found]})

    if args.needed:
        return _models_needed(conn)
    if args.verify:
        return _models_verify(conn, args.verify)
    if args.source:
        name, uri = args.source
        matches = [m for m in found if name.lower() in m.name.lower()]
        if len(matches) != 1:
            print(f"--source must match exactly one model; {name!r} matches "
                  f"{', '.join(m.name for m in matches) or 'nothing'}", file=sys.stderr)
            return 2
        db.set_model_source(conn, matches[0].id, uri)
        print(f"{matches[0].name}\n  obtainable from {uri}")
        return 0

    if args.name:
        wanted, alias = args.name
        row = db.find_model(conn, wanted)
        if row is None:
            print(f"no model matches {wanted!r}; `dyp models` lists them", file=sys.stderr)
            return 2
        db.set_alias(conn, row["id"], alias)
        print(f"{_shorten(row['name'])} is now also {alias!r}")
        print(f"  dyp embed --model {alias}")
        return 0

    if args.truncate:
        from dyprys.library import truncate_model

        if len(found) != 1:
            print("--truncate needs exactly one model in the index", file=sys.stderr)
            return 2
        target = found[0]
        print(f"this would keep the first {args.truncate} of {target.dim} dimensions "
              f"of {target.name}.")
        print("Sound ONLY for a model trained with Matryoshka representation learning,")
        print("where the dimensions are ordered by importance — EmbeddingGemma is, most")
        print("models are not, and on one that is not this destroys the index silently.")
        print("Measured on 117 books: 768 -> 512 costs nothing (paired p = 0.73);")
        print("768 -> 256 halves storage again and costs recall@1 but not recall@5.")
        print("\nthe routing profile is dropped — rerun `dyp route` afterwards.")
        if not args.yes:
            print("\nre-run with --yes to go ahead.", file=sys.stderr)
            return 1

        def show(done, total):
            print(f"\r  {done:,}/{total:,} vectors rewritten ", end="", file=sys.stderr, flush=True)

        try:
            before, after = truncate_model(conn, directory, target.id, args.truncate, show)
        except ValueError as why:
            print(f"\n{why}", file=sys.stderr)
            return 1
        print(file=sys.stderr)
        db.record_event(conn, "truncate",
                        f"{_shorten(target.name)} cut from {target.dim} to "
                        f"{args.truncate} dimensions")
        print(f"{target.name} is now {args.truncate} dimensions")
        if after:
            print(f"  {_size(before)} → {_size(after)} on disk  ({before / after:.1f}x smaller)")
        return 0

    if args.quantise:
        from dyprys.library import quantise_model

        matches = [m for m in found if args.quantise.lower() in m.name.lower()]
        if len(matches) != 1:
            which = ", ".join(m.name for m in matches) or "nothing"
            print(f"--quantise must match exactly one model; {args.quantise!r} "
                  f"matches {which}", file=sys.stderr)
            return 2
        target = matches[0]

        def show(done, total):
            print(f"\r  {done:,}/{total:,} vectors converted ", end="", file=sys.stderr, flush=True)

        try:
            before, after = quantise_model(conn, directory, target.id, show)
        except ValueError as why:
            print(f"\n{why}", file=sys.stderr)
            return 1
        print(file=sys.stderr)
        db.record_event(conn, "quantise", f"{_shorten(target.name)} converted to int8")
        print(f"{target.name} is now int8")
        print(f"  {_size(before)} → {_size(after)} on disk"
              f"  ({before / after:.1f}x smaller)" if after else "")
        return 0

    if args.drop:
        matches = [m for m in found if args.drop.lower() in m.name.lower()]
        if len(matches) != 1:
            which = ", ".join(m.name for m in matches) or "nothing"
            print(f"--drop must match exactly one model; {args.drop!r} matches {which}",
                  file=sys.stderr)
            return 2
        victim = matches[0]
        print(f"this would remove {victim.name}")
        print(f"  {victim.embedded:,} embedded chunks and {_size(victim.bytes_on_disk)} on disk")
        print("  books, chunks and the BM25 index are untouched")
        if not args.yes:
            print("\nre-run with --yes to go ahead. Re-embedding this model would take "
                  f"about {victim.embedded / 6.6 / 3600:.1f} h.", file=sys.stderr)
            return 1
        for path in drop_model(conn, directory, victim.id):
            print(f"removed {path.name}")
        print(f"dropped {victim.name}")
        return 0

    print(f"{'model':<34} {'dim':>5} {'store':>6} {'embedded':>12} {'cover':>6} "
          f"{'disk':>9} {'routing':>18}")
    for m in found:
        if not m.centroid_books:
            routing = "not built"
        elif m.stale_books:
            routing = f"{m.stale_books} book(s) stale"
        else:
            routing = f"{m.centroid_books} books profiled"
        print(f"{_shorten(m.name):<34} {m.dim:>5} {m.quantisation:>6} {m.embedded:>12,} "
              f"{m.coverage:>6.0%} {_size(m.bytes_on_disk):>9} {routing:>18}")
        # What you can type for --model. A name nobody can discover is no better
        # than no name, and the full one above is a hash you would not retype.
        handles = [m.alias] if m.alias else []
        handles.append(m.name.split("@")[0][-28:] if "@" in m.name else m.name)
        print(f"    --model  {'  or  '.join(repr(h) for h in handles)}"
              f"{'' if m.alias else '   (dyp models --name … ALIAS for a shorter one)'}")
        if m.file_path:
            here = "" if Path(m.file_path).exists() else "   MISSING"
            print(f"    weights  {m.file_path}{here}")
        elif not m.alias:
            print("    weights  not recorded yet — pass --model once and it is remembered")
        if m.failures or m.carries:
            extra = []
            if m.failures:
                extra.append(f"{m.failures:,} failed chunks")
            if m.carries:
                extra.append(f"{m.carries:,} vectors waiting to be carried")
            print(f"{'':<34} {', '.join(extra)}")

    _print_optional_models(conn)

    sparse = sum(m.vectors_apparent for m in found), sum(m.vectors_actual for m in found)
    if sparse[0] > sparse[1] * 1.2:
        print(f"\nvector files are sparse: {_size(sparse[0])} addressable, "
              f"{_size(sparse[1])} actually on disk")
    return 0


def _eval(conn, directory, args) -> int:
    loaded = _load_model(conn, directory, args.model)
    if loaded is None:
        return 2
    embedder, model_id, store = loaded
    from dyprys.evaluate import evaluate, lexical_safety, load_questions

    if not args.questions.exists():
        print(f"no question set at {args.questions}", file=sys.stderr)
        return 2
    questions = load_questions(args.questions)

    from dyprys.search import flat_search, scanned_fraction, scope_books

    books = scope_books(conn, args.collection) if args.collection else None
    if args.collection and not books:
        print(f"no book matches {args.collection!r}", file=sys.stderr)
        return 1

    router, problem = _router(conn, directory, embedder, model_id, args.route, books)
    if problem:
        return 2

    reranker, missing = _load_reranker(args, conn)
    if missing:
        return 2
    expander, missing = _load_expander(args, conn)
    if missing:
        print(missing, file=sys.stderr)
        return 2

    # --rerank N implies depth N; --depth N asks for the same shortlist with no
    # reordering, which is the arm that separates the two.
    depth = args.rerank or getattr(args, "depth", 0) or 0
    modes = ("vector", "lexical", "hybrid") if args.compare else (args.mode,)
    results = {}
    for mode in modes:
        run = _searcher(conn, store, embedder, model_id, mode, books, router,
                        reranker, depth, args.dedupe, expander)
        touched: list[float] = []

        def measured(question, k, _run=run):
            hits = _run(question, k)
            if router:
                touched.append(scanned_fraction(conn, model_id, router.last["books"]))
            return hits

        # With routing on, also run the search unrouted, so the report can say
        # whether stage 1 kept the book flat search itself chose. That is the
        # question routing recall was meant to answer and cannot, once the
        # answer is in most of the library.
        #
        # Deliberately *without* the reranker, for two reasons. It asks which
        # book stage 1 should have kept, which is a question about retrieval;
        # reranking the comparison moved the figure 69/110 -> 55/110 by changing
        # which passage led the flat list, so the number stopped meaning what its
        # label says. And it is the whole diagnostic's cost: reranking it doubled
        # a routed --rerank eval, 50 minutes to 112.
        unrouted = _searcher(conn, store, embedder, model_id, mode, books, None,
                             None, depth, args.dedupe,
                             expander) if router else None
        rep = evaluate(
            conn, store, embedder, model_id, questions, k=args.k, search_fn=measured,
            scanned=scanned_fraction(conn, model_id, books),
            route_fn=(lambda q: router(q, embedder.embed_query(q))) if router else None,
            flat_fn=(lambda q: unrouted(q, args.k)) if unrouted else None,
        )
        if touched:
            rep.scanned = sum(touched) / len(touched)
        lex = lexical_safety(conn, store, embedder, model_id, samples=args.lexical,
                             k=args.k, search_fn=run)
        results[mode] = (rep, lex)

    if args.compare:
        first = next(iter(results.values()))[0]
        kinds = [k for k in ("lookup", "descriptive", "mechanism", "oblique")
                 if first.by_kind.get(k, None) and first.by_kind[k].answerable]
        head = "".join(f"{k[:9]:>10}" for k in kinds)
        print(f"{'mode':<9} {'recall@1':>9} {'recall@' + str(args.k):>9} {'lexical':>9}"
              f" {'scanned':>8}   recall@1 by kind:{head}")
        for mode, (rep, (hits, asked)) in results.items():
            per = "".join(
                f"{rep.by_kind[k].hit_at_1 / rep.by_kind[k].answerable:>10.0%}" for k in kinds
            )
            print(f"{mode:<9} {rep.hit_at_1:>4}/{rep.answerable:<4} {rep.hit_at_5:>4}/"
                  f"{rep.answerable:<4} {hits:>4}/{asked:<4} {rep.scanned:>7.1%}"
                  f"                    {per}")
        return 0

    report, (hits, asked) = results[args.mode]

    print(f"questions          {report.asked:>6}")
    print(f"answerable         {report.answerable:>6}  (term present in the embedded corpus)")
    if report.unanswerable:
        shown = ", ".join(sorted(set(report.unanswerable))[:6])
        print(f"  excluded: {shown}{' …' if len(set(report.unanswerable)) > 6 else ''}")
    print()
    print(f"answer recall @1   {report.hit_at_1:>6}/{report.answerable}  {report.recall_at_1:.0%}")
    print(f"answer recall @{args.k}   {report.hit_at_5:>6}/{report.answerable}  {report.recall_at_5:.0%}")
    print(f"lexical safety     {hits:>6}/{asked}  exact phrase finds its own passage")
    if report.routed_asked:
        print(f"routing recall     {report.routed_ok:>6}/{report.routed_asked}  "
              f"a book holding the answer survived stage 1")
        # Never the recall alone. On a corpus where the answer is everywhere,
        # picking books at random scores nearly as well, and for a long time
        # this measure was quoted as though it proved something.
        print(f"  ...by chance     {report.routed_chance:>6.1%}  "
              f"the same, for books picked at random")
        if report.kept_flat_asked:
            print(f"  ...kept the best {report.kept_flat_top:>6}/{report.kept_flat_asked}  "
                  f"routed set held the book flat search chose")
    if report.answer_books:
        print(f"answer redundancy  {report.redundancy:>6}  "
              f"median books holding the answer, of {len(report.answer_books) and ''}"
              f"{conn.execute('SELECT COUNT(*) FROM books').fetchone()[0]}")
    embedded_now = conn.execute(
        "SELECT COALESCE(SUM(n_embedded), 0) FROM segment_progress WHERE model_id = ?",
        (model_id,)).fetchone()[0]
    print(f"vectors scanned    {report.scanned:>6.1%}  "
          f"= {report.scanned * embedded_now:,.0f} of {embedded_now:,} chunks")
    print("  (a share of the corpus only compares across runs on the *same* corpus:")
    print("   adding unrelated books shrinks the fraction without shrinking the work)")
    print(f"query time         {report.seconds / max(report.answerable, 1) * 1000:>6.1f} ms each")
    def breakdown(title, tallies, order=None):
        if len(tallies) < 2:
            return
        print(f"\n{title:<18} {'answerable':>10} {'recall@1':>9} {'recall@' + str(args.k):>9}")
        for name in (order or sorted(tallies)):
            t = tallies.get(name)
            if not t or not t.answerable:
                continue
            print(f"  {name:<16} {t.answerable:>10} "
                  f"{t.hit_at_1 / t.answerable:>8.0%} {t.hit_at_5 / t.answerable:>9.0%}")

    breakdown("by question kind", report.by_kind,
              order=("lookup", "descriptive", "mechanism", "oblique"))
    breakdown("by measured overlap", report.by_overlap,
              order=("shares wording", "shares little"))

    if report.misses_at_1:
        # Every question whose answer was not ranked first, not a sample: two
        # runs are compared by diffing these lists, and a truncated list would
        # make the comparison quietly wrong rather than obviously incomplete.
        print(f"\nnot first ({len(report.misses_at_1)}):")
        for question, term in report.misses_at_1:
            print(f"  {term:<22} {question[:64]}")
    if report.misses:
        # In full, and for the same reason as the list above: reranking moves
        # passages from rank 6-20 into the top 5, so recall@5 is a real measure
        # of it, and a list cut off at 8 makes the diff between two arms wrong
        # rather than obviously short.
        print(f"\nmissed ({len(report.misses)}):")
        for question, term in report.misses:
            print(f"  {term:<22} {question[:64]}")
    return 0


def _check(conn, deep: bool, as_json: bool = False) -> int:
    """Report drift and outstanding work. Never fixes anything by itself."""
    report = survey(conn, deep=deep)

    if as_json:
        from dyprys.routing import is_built, stale_books

        def routing_for(name):
            row = conn.execute("SELECT id FROM models WHERE name = ?", (name,)).fetchone()
            if not row or not is_built(conn, row["id"]):
                return {"built": False, "stale_books": None}
            return {"built": True, "stale_books": stale_books(conn, row["id"])}

        return _emit_json({
            "deep": deep,
            "sources": report.sources,
            "drift": {
                "missing": report.drift.missing,
                "changed": report.drift.changed,
                "intact": report.drift.intact,
                "clean": report.drift.clean,
            },
            "live_chunks": report.live_chunks,
            "dead_chunks": report.dead_chunks,
            "lexical_chunks": report.lexical_chunks,
            "lexical_complete": report.lexical_chunks == report.live_chunks,
            "garbled": [
                {"title": g.title, "chunks": g.chunks, "p90_token": g.p90_token}
                for g in report.garbled],
            "models": [
                {"name": m.name, "dim": m.dim, "embedded": m.embedded,
                 "to_copy": m.to_copy, "to_embed": m.to_embed, "failed": m.failed,
                 "outstanding": m.outstanding,
                 "coverage": (m.embedded / (m.embedded + m.outstanding))
                 if (m.embedded + m.outstanding) else 0.0,
                 "routing": routing_for(m.name)}
                for m in report.models],
        })

    how = "re-hashed" if deep else "checked by size and mtime"
    print(f"{report.sources:,} source files, {how}")
    if report.drift.clean:
        print("  all intact")
    else:
        print(f"  {report.drift.intact:,} intact")
        for label, paths in (("missing", report.drift.missing), ("changed", report.drift.changed)):
            if paths:
                print(f"  {len(paths):,} {label}")
                for path in paths[:5]:
                    print(f"      {path}")
                if len(paths) > 5:
                    print(f"      … and {len(paths) - 5:,} more")

    print(f"\n{report.live_chunks:,} live chunks")
    if report.dead_chunks:
        print(f"{report.dead_chunks:,} superseded by re-ingest, reclaimable by compaction")
    if report.lexical_chunks == report.live_chunks and report.live_chunks:
        print(f"{report.lexical_chunks:,} in the BM25 index  (complete)")
    else:
        missing = report.live_chunks - report.lexical_chunks
        print(f"{report.lexical_chunks:,} in the BM25 index  "
              f"({missing:,} missing — run `dyp lexical`)")

    if report.garbled:
        total = sum(g.chunks for g in report.garbled)
        print(f"\n{len(report.garbled)} book(s) whose text has no word boundaries — "
              f"{total:,} chunks", file=sys.stderr)
        for g in report.garbled[:5]:
            print(f"      {g.chunks:>6,} chunks, longest words ~{g.p90_token}  {g.title[:48]}",
                  file=sys.stderr)
        if len(report.garbled) > 5:
            print(f"      … and {len(report.garbled) - 5} more", file=sys.stderr)
        print("  the PDF's font map defeated the extractor. Embedding these costs "
              "the same as", file=sys.stderr)
        print("  real text and can never match a query; re-extract them or "
              "`dyp remove` them.", file=sys.stderr)

    if not report.models:
        print("\nno embedding model registered yet")
    else:
        print()
        for m in report.models:
            # Against the chunking this model works on, not the library. With
            # two chunkings the library total is a number this model can never
            # reach, so a finished model reported itself part-done for ever.
            whole = m.embedded + m.outstanding or report.live_chunks
            done = f"{m.embedded / whole:.1%}" if whole else "--"
            split = (f", {m.chunk_target:,}-byte chunks"
                     if m.chunk_target and len(db.chunkings(conn)) > 1 else "")
            print(f"{_shorten(m.name)}  (dim {m.dim}{split})")
            print(f"    embedded   {m.embedded:>12,}  {done}")
            if m.to_copy:
                print(f"    to copy    {m.to_copy:>12,}  vectors survive an edit — near-free")
            if m.to_embed:
                rate = db.observed_rate(conn, m.id)
                if rate:
                    print(f"    to embed   {m.to_embed:>12,}  ~{m.to_embed / rate / 3600:.1f} h "
                          f"at {rate:.1f} chunks/s (this machine's median)")
                else:
                    print(f"    to embed   {m.to_embed:>12,}  no runs yet, so no rate to "
                          f"estimate from")
            if m.failed:
                print(f"    failed     {m.failed:>12,}  recorded, retryable")
            if not m.outstanding and not m.failed:
                print("    up to date")

    from dyprys.routing import is_built, stale_books
    for m in report.models:
        row = conn.execute("SELECT id FROM models WHERE name = ?", (m.name,)).fetchone()
        if row and is_built(conn, row["id"]):
            stale = stale_books(conn, row["id"])
            note = f"{stale} book(s) changed since — rerun `dyp route`" if stale else "current"
            print(f"\nrouting profile   {note}")
        elif row:
            print("\nrouting profile   not built — run `dyp route` for two-stage search")

    if report.drift.changed:
        print("\nrun `dyp add` on the changed files to re-chunk them;")
        print("unchanged passages keep their vectors.")
    if report.drift.missing:
        print("\nmissing files are reported, never removed. if one was moved rather")
        print("than deleted, `dyp add` on its new location relocates it for free.")
    return 0


def _status(conn, as_json: bool = False) -> int:
    books, sources, chunks, text_bytes = conn.execute(
        "SELECT (SELECT COUNT(*) FROM books), "
        "       (SELECT COUNT(*) FROM sources), "
        "       (SELECT COUNT(*) FROM chunks), "
        "       (SELECT COALESCE(SUM(size_bytes), 0) FROM sources)"
    ).fetchone()

    chunkings_rows = conn.execute(
        "SELECT ch.id, ch.target, ch.overlap, "
        "       COALESCE(SUM(seg.chunk_count), 0) AS n "
        "FROM chunkings ch LEFT JOIN segments seg ON seg.chunking_id = ch.id "
        "GROUP BY ch.id ORDER BY ch.id"
    ).fetchall()
    model_rows = conn.execute(
        "SELECT m.id, m.name, m.dim, COALESCE(SUM(p.n_embedded), 0) AS done "
        "FROM models m LEFT JOIN segment_progress p ON p.model_id = m.id "
        "GROUP BY m.id ORDER BY m.id"
    ).fetchall()
    from dyprys.compact import interrupted
    carries = conn.execute("SELECT COUNT(*) FROM chunk_carry").fetchone()[0]
    fails = conn.execute("SELECT COUNT(*) FROM chunk_failures").fetchone()[0]

    if as_json:
        return _emit_json({
            "books": books, "sources": sources, "chunks": chunks,
            "text_bytes": text_bytes,
            "chunkings": [
                {"id": c["id"], "target": c["target"], "overlap": c["overlap"],
                 "chunks": c["n"]} for c in chunkings_rows],
            "models": [
                {"name": m["name"], "dim": m["dim"], "embedded": m["done"],
                 "coverage": (m["done"] / chunks) if chunks else 0.0}
                for m in model_rows],
            "pending_carries": carries,
            "failed_chunks": fails,
            "compaction_interrupted": interrupted(conn),
        })

    print(f"books      {books:>12,}")
    print(f"sources    {sources:>12,}")
    print(f"text       {_size(text_bytes):>12}")
    print(f"chunks     {chunks:>12,}")

    if len(chunkings_rows) > 1:
        print(f"\n{'chunking':<20} {'chunks':>12}")
        for c in chunkings_rows:
            label = f"{c['target']}B / {c['overlap']}B overlap"
            print(f"{label:<20} {c['n']:>12,}")

    # Embedding progress is per model: a library is routinely complete under one
    # model and untouched under another.
    if not model_rows:
        print("\nno embedding model registered yet")
    else:
        print(f"\n{'model':<34} {'dim':>5} {'embedded':>12}")
        for m in model_rows:
            pct = f"{m['done'] / chunks:.1%}" if chunks else "--"
            print(f"{_shorten(m['name']):<34} {m['dim']:>5} {m['done']:>12,} {pct:>7}")

    if interrupted(conn):
        print("\na compaction stopped part-way through — run `dyp compact` to finish it")

    if carries:
        print(f"\n{carries:,} vectors waiting to be carried over from edited files")
    if fails:
        print(f"{fails:,} chunks failed to embed (recorded, retryable)")

    # The last thing done to this index, so the journal is discoverable without
    # already knowing it exists — and so "why does this look like that" has an
    # answer on the screen you were already reading.
    recent = conn.execute(
        "SELECT at, action, detail FROM events ORDER BY id DESC LIMIT 2").fetchall()
    if recent:
        print("\nlast changes")
        for e in recent:
            print(f"  {_ago(e['at']):>9}  {e['action']:<9} {e['detail']}")
        print("  `dyp history` for more")
    _say_next(conn)
    return 0


def _emit_json(payload) -> int:
    """Print a payload as JSON and succeed.

    One place so every inspection command's `--json` looks the same. `default=str`
    turns a Path into its string form rather than raising.
    """
    import json as _json

    print(_json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


def _shorten(name: str, width: int = 34) -> str:
    """Model names are long paths or hub ids; keep the identifying tail."""
    return name if len(name) <= width else "…" + name[-(width - 1) :]


if __name__ == "__main__":
    raise SystemExit(main())
