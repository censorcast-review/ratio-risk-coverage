"""Build a compact, hash-checked submission ZIP below 95,000,000 bytes.

This packaging step does not modify source evidence or the manuscript. It runs
after the integrated manuscript has been built. Keep the full archive separately.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import zipfile

HERE = Path(__file__).resolve().parent
TOOLS = HERE / 'submission_tools'
TEXT = {'.json', '.jsonl', '.csv', '.md', '.py', '.sh', '.yml', '.yaml', '.toml', '.sha256', '.log', '.ipynb'}
PAPER = {'.tex', '.bib', '.sty', '.bst', '.pdf', '.png', '.jpg', '.jpeg', '.svg'}
V13_EXCLUDE_DIRS = {'raw', 'delivery', 'prior_report', 'visual_qa', '__pycache__', '.git'}
V13_EXCLUDE_FILES = {'build_report.py', 'package_release.py', 'RELEASE_MANIFEST.json', 'README.md'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def imported(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select(relative, size):
    """Return None for an original included file, otherwise an explicit reason."""
    p = Path(relative)
    if any(part in {'__pycache__', '.git'} for part in p.parts):
        return 'cache or repository metadata'
    if p.suffix in {'.pyc', '.writing'} or '.openai-download-' in p.name or size == 0:
        return 'temporary or empty file'
    if p.parts[0] in {'data_parts', 'reproduction_outputs', 'delivery'}:
        return 'full archive parts or regenerated output'
    if relative in {'MANIFEST.json', 'REVIEW_LINK.json', '.gitignore', 'README.md'}:
        return 'full archive metadata or replaced entrypoint'
    if relative == 'docs/revision_v9/PUBLIC_REPOSITORY_SNAPSHOT.json' or relative.startswith('docs/revision_v9/REPOSITORY_'):
        return 'historical public synchronization metadata; unrelated to scientific replay'
    if p.parts[0] == 'paper':
        if p.suffix not in PAPER:
            return 'paper build byproduct'
        # Preserve every figure, but only the final manuscript PDF at paper root.
        if len(p.parts) == 2 and p.suffix == '.pdf' and p.name != 'CENSORCAST_ICLR_2027_Final.pdf':
            if p.stem in {'matched_history', 'revision_v8_neartie'}:
                return None
            return 'historical or intermediate manuscript PDF'
        return None
    if p.parts[0] == 'code':
        return None if p.suffix in TEXT or p.name.startswith('requirements') else 'non-code artifact'
    if len(p.parts) == 1 and p.name.startswith(('requirements', 'LICENSE', 'NOTICE')):
        return None
    if p.parts[0] == 'docs':
        return None if p.suffix in TEXT else 'historical rendered/editing asset'
    if p.parts[0] == 'evidence':
        if p.parts[1] == 'revision_v13_nonwape':
            tail = Path(*p.parts[2:])
            if any(part in V13_EXCLUDE_DIRS for part in tail.parts):
                return 'v13 raw, old delivery, historical report, or preview'
            if tail.as_posix() in V13_EXCLUDE_FILES or tail.suffix in {'.zip', '.rar'}:
                return 'v13 historical entrypoint or recursive archive'
            return None
        if p.suffix in TEXT:
            return None
        if p.parts[1] == 'revision_v12_external':
            return None if p.suffix in {'.npz', '.txt'} else 'raw or unsupported external asset'
        if p.parts[1] in {'matched_censoring', 'revision_v12_seeds'}:
            return 'large M5 source assets replaced by labelled primary item statistics; original source hashes retained'
        if p.suffix == '.npz' and size <= 10_000_000:
            return None
        return 'large model/prediction asset or raw source data'
    if p.parts[0] in {'legacy', 'forecast_archive'}:
        if 'inputs' in p.parts or ('paper' in p.parts and p.suffix not in {'.py', '.md'}):
            return 'historical raw/development input or manuscript copy'
        return None if p.suffix in TEXT else 'historical model, large array or rendered artifact'
    return 'outside declared compact scope'


def cached_derivation_valid(source, cache):
    f = cache / 'DERIVATION.json'
    if not f.is_file():
        return False
    m = read(f)
    if m.get('derivation_code_sha256') != sha(TOOLS / 'derive_m5_primary.py'):
        return False
    return (all((source / rel).is_file() and sha(source / rel) == rec['sha256']
                for rel, rec in m['source_files'].items())
            and all((cache / name).is_file() and sha(cache / name) == rec['sha256']
                    for name, rec in m['files'].items()))


def copy_original(source, relative, stage, included):
    src, dst = source / relative, stage / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    assert sha(src) == sha(dst), relative
    included[relative] = {'sha256': sha(dst), 'bytes': dst.stat().st_size, 'kind': 'original-byte-copy'}


def scan_text(stage, extra_patterns):
    # Do not write private search patterns or matched text into this supplement.
    # Dataset/third-party references can legitimately contain contact addresses;
    # they are flagged for manual review rather than automatically edited.
    patterns = [('personal_home_path', re.compile(r'(?:/Users/|/home/|[A-Za-z]:\\Users\\)[^\s/\\]+')),
                ('contact_address', re.compile(r'(?<![\w.])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.])'))]
    hits, blocked = [], []
    for f in sorted(stage.rglob('*')):
        if not f.is_file() or f.suffix not in TEXT | {'.txt', '.tex', '.bib', '.sty', '.bst'}:
            continue
        text = f.read_text(errors='replace')
        rel = f.relative_to(stage).as_posix()
        for kind, pattern in patterns:
            if pattern.search(text):
                hits.append({'file': rel, 'category': kind})
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in extra_patterns):
            blocked.append(rel)
    if blocked:
        raise RuntimeError('Private anonymity pattern found in: ' + ', '.join(blocked))
    return {'scope': 'text/code/TeX scan; binary model and PDF metadata require separate review',
            'generic_home_and_email_hits_for_manual_review': hits,
            'private_pattern_matches': 0, 'extra_private_patterns_supplied': bool(extra_patterns),
            'source_files_changed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=HERE / 'CENSORCAST')
    parser.add_argument('--v13-source', type=Path)
    parser.add_argument('--output', type=Path, default=HERE / 'delivery/CENSORCAST_v13_Integrated_Submission_Supplement.zip')
    parser.add_argument('--stage', type=Path, default=HERE / 'submission_stage/CENSORCAST')
    parser.add_argument('--cache', type=Path, default=HERE / 'submission_cache/m5_primary')
    parser.add_argument('--max-bytes', type=int, default=95_000_000)
    parser.add_argument('--deny-pattern-file', type=Path,
                        help='Optional private regex list; never copied into the supplement')
    parser.add_argument('--skip-compact-verification', action='store_true',
                        help='Development only; final releases must omit this flag')
    args = parser.parse_args()
    source, stage, target = args.source.resolve(), args.stage.resolve(), args.output.resolve()
    if source == stage or source in stage.parents or stage in source.parents:
        raise ValueError('Stage must be separate from the immutable source tree')
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    included, excluded = {}, []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(source).as_posix()
        reason = select(rel, path.stat().st_size)
        if reason:
            excluded.append({'path': rel, 'bytes': path.stat().st_size, 'reason': reason})
        else:
            copy_original(source, rel, stage, included)
    v13 = source / 'evidence/revision_v13_nonwape'
    if not v13.is_dir():
        if args.v13_source is None:
            raise RuntimeError('Copy v13 into evidence/revision_v13_nonwape or provide --v13-source')
        for path in sorted(args.v13_source.rglob('*')):
            if not path.is_file():
                continue
            rel = path.relative_to(args.v13_source).as_posix()
            destrel = f'evidence/revision_v13_nonwape/{rel}'
            reason = select(destrel, path.stat().st_size)
            if reason:
                excluded.append({'path': destrel, 'bytes': path.stat().st_size, 'reason': reason})
            else:
                dst = stage / destrel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst)
                assert sha(path) == sha(dst)
                included[destrel] = {'sha256': sha(dst), 'bytes': dst.stat().st_size, 'kind': 'original-byte-copy'}
    if not cached_derivation_valid(source, args.cache):
        imported('derive_m5_primary', TOOLS / 'derive_m5_primary.py').derive(source, args.cache)
    derived = stage / 'submission_derived/m5_primary'
    shutil.copytree(args.cache, derived)
    for name in ('derive_m5_primary.py', 'verify_compact_m5.py'):
        shutil.copy2(TOOLS / name, stage / 'code' / name)
    (stage / 'submission_tools').mkdir()
    for name in ('derive_m5_primary.py', 'verify_compact_m5.py', 'SUBMISSION_README.md'):
        shutil.copy2(TOOLS / name, stage / 'submission_tools' / name)
    shutil.copy2(__file__, stage / 'build_submission_bundle.py')
    shutil.copy2(TOOLS / 'SUBMISSION_README.md', stage / 'README.md')
    if (source / 'README.md').is_file():
        shutil.copy2(source / 'README.md', stage / 'ARCHIVE_README.md')
    if not args.skip_compact_verification:
        imported('verify_compact_m5', TOOLS / 'verify_compact_m5.py').verify(
            stage, stage / 'submission_derived/COMPACT_M5_VERIFICATION.json')
    patterns = ([] if args.deny_pattern_file is None else
                [line for line in args.deny_pattern_file.read_text().splitlines() if line and not line.startswith('#')])
    anonymity = scan_text(stage, patterns)
    (stage / 'submission_derived/TEXT_ANONYMITY_SCAN.json').write_text(json.dumps(anonymity, indent=2) + '\n')
    for path in sorted(stage.rglob('*')):
        if path.is_file():
            rel = path.relative_to(stage).as_posix()
            if rel not in included:
                included[rel] = {'sha256': sha(path), 'bytes': path.stat().st_size,
                                 'kind': 'packaging-derived-or-generated'}
    manifest = {'schema': 'compact-submission-v1',
                'scope': 'integrated manuscript and compact scientific replay supplement; not the complete historical archive',
                'max_zip_bytes': args.max_bytes,
                'compact_m5_verification': 'SKIPPED-DEVELOPMENT' if args.skip_compact_verification else 'PASS',
                'source_evidence_modified': False, 'original_included_files': len([v for v in included.values() if v['kind'] == 'original-byte-copy']),
                'files': included, 'excluded': excluded}
    (stage / 'SUBMISSION_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.writing')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                archive.write(path, 'CENSORCAST/' + path.relative_to(stage).as_posix())
    size = temporary.stat().st_size
    if size >= args.max_bytes:
        raise RuntimeError(f'Compact ZIP {size} bytes exceeds safety limit {args.max_bytes}; inspect exclusions without changing scientific results')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for rel, rec in included.items():
            assert hashlib.sha256(archive.read('CENSORCAST/' + rel)).hexdigest() == rec['sha256'], rel
        assert archive.read('CENSORCAST/paper/CENSORCAST_ICLR_2027_Final.pdf') == (source / 'paper/CENSORCAST_ICLR_2027_Final.pdf').read_bytes()
    temporary.replace(target)
    report = {'status': 'PASS' if not args.skip_compact_verification else 'DEVELOPMENT',
              'zip_bytes': size, 'zip_sha256': sha(target), 'safety_limit_bytes': args.max_bytes,
              'members': len(included) + 1, 'all_member_hashes_and_crc': 'PASS',
              'manuscript_pdf_identical_to_source': True,
              'original_included_files': manifest['original_included_files'], 'excluded_files': len(excluded),
              'anonymous_text_potential_hits': anonymity['generic_home_and_email_hits_for_manual_review'],
              'complete_historical_archive': False}
    target.with_suffix('.CHECK.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    for env in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[env] = '1'
    main()
