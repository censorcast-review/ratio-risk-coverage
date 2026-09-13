"""Read a locally frozen Mulan ARFF/XML archive; performs no network access.

The caller must freeze its protocol and this module before downloading raw data.
Archive acquisition, role splitting, learner fitting, and E opening are caller-owned.
"""
from __future__ import annotations
import ctypes
import ctypes.util
import csv
import hashlib
import io
import json
import pathlib
import re
import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

EXPECTED = {
    'bibtex': {'n_features': 1836, 'n_labels': 159, 'n_total': 7395},
    'mediamill': {'n_features': 120, 'n_labels': 101, 'n_total': 43907,
                  'n_train': 30993, 'n_test': 12914},
}


def _archive_text_members(path: str | pathlib.Path) -> dict[str, bytes]:
    """Extract only regular ARFF/XML members into memory via system libarchive.

    No shell command, redirect, path writing, or external extraction utility is used.
    Duplicate names, encryption, unknown sizes, and oversized members abort.
    """
    libname = ctypes.util.find_library('archive')
    if not libname:
        raise RuntimeError('LIBARCHIVE_UNAVAILABLE')
    lib = ctypes.CDLL(libname)
    signatures = {
        'archive_read_new': (ctypes.c_void_p, []),
        'archive_read_support_filter_all': (ctypes.c_int, [ctypes.c_void_p]),
        'archive_read_support_format_all': (ctypes.c_int, [ctypes.c_void_p]),
        'archive_read_open_filename': (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]),
        'archive_read_next_header': (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]),
        'archive_entry_pathname': (ctypes.c_char_p, [ctypes.c_void_p]),
        'archive_entry_size': (ctypes.c_int64, [ctypes.c_void_p]),
        'archive_entry_filetype': (ctypes.c_uint, [ctypes.c_void_p]),
        'archive_entry_is_encrypted': (ctypes.c_int, [ctypes.c_void_p]),
        'archive_read_data': (ctypes.c_ssize_t, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]),
        'archive_read_data_skip': (ctypes.c_int, [ctypes.c_void_p]),
        'archive_error_string': (ctypes.c_char_p, [ctypes.c_void_p]),
        'archive_read_free': (ctypes.c_int, [ctypes.c_void_p]),
    }
    for name, (rtype, args) in signatures.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = rtype, args
    handle = lib.archive_read_new()
    if not handle:
        raise RuntimeError('LIBARCHIVE_ALLOCATION_FAILED')
    out: dict[str, bytes] = {}
    total = 0
    def fail(where: str):
        message = lib.archive_error_string(handle)
        detail = message.decode('utf-8', 'replace') if message else 'no error detail'
        raise ValueError(f'{where}: {detail}')
    try:
        if lib.archive_read_support_filter_all(handle) < 0: fail('ARCHIVE_FILTER')
        if lib.archive_read_support_format_all(handle) < 0: fail('ARCHIVE_FORMAT')
        if lib.archive_read_open_filename(handle, str(path).encode(), 65536) < 0:
            fail('ARCHIVE_OPEN')
        while True:
            entry = ctypes.c_void_p()
            status = lib.archive_read_next_header(handle, ctypes.byref(entry))
            if status == 1:  # ARCHIVE_EOF
                break
            if status < 0:
                fail('ARCHIVE_HEADER')
            name_raw = lib.archive_entry_pathname(entry)
            if not name_raw:
                raise ValueError('ARCHIVE_MEMBER_WITHOUT_NAME')
            name = name_raw.decode('utf-8', 'strict').replace('\\', '/')
            p = pathlib.PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError(f'UNSAFE_ARCHIVE_MEMBER: {name}')
            if p.suffix.lower() not in {'.arff', '.xml'}:
                if lib.archive_read_data_skip(handle) < 0: fail('ARCHIVE_SKIP')
                continue
            if lib.archive_entry_filetype(entry) != 0o100000:
                raise ValueError(f'NONREGULAR_DATA_MEMBER: {name}')
            if lib.archive_entry_is_encrypted(entry) != 0:
                raise ValueError(f'ENCRYPTED_DATA_MEMBER: {name}')
            size = int(lib.archive_entry_size(entry))
            if size < 0 or size > 512 * 1024 * 1024 or total + size > 1024**3:
                raise ValueError(f'ARCHIVE_SIZE_LIMIT: {name}, {size}')
            if name in out:
                raise ValueError(f'DUPLICATE_MEMBER_NAME: {name}')
            chunks, count = [], 0
            buf = ctypes.create_string_buffer(1024 * 1024)
            while True:
                got = int(lib.archive_read_data(handle, buf, len(buf)))
                if got == 0:
                    break
                if got < 0:
                    fail('ARCHIVE_READ')
                count += got
                if count > size:
                    raise ValueError(f'ARCHIVE_SIZE_MISMATCH: {name}')
                chunks.append(buf.raw[:got])
            if count != size:
                raise ValueError(f'ARCHIVE_SIZE_MISMATCH: {name}, {count} != {size}')
            out[name] = b''.join(chunks)
            total += size
    finally:
        lib.archive_read_free(handle)
    return out


def xml_labels(data: bytes) -> list[str]:
    root = ET.fromstring(data)
    names = [node.attrib['name'] for node in root.iter()
             if node.tag.rsplit('}', 1)[-1] == 'label']
    if not names or len(names) != len(set(names)):
        raise ValueError('XML_EMPTY_OR_DUPLICATE_LABELS')
    return names


def _attr_name_type(line: str) -> tuple[str, str]:
    rest = re.sub(r'^@attribute\s+', '', line, count=1, flags=re.I).strip()
    if rest[:1] in {'"', "'"}:
        quote = rest[0]
        m = re.match(r"^([\"'])(.*?)\1\s+(.+)$", rest)
        if not m:
            raise ValueError(f'ARFF_BAD_ATTRIBUTE: {line[:150]}')
        name, kind = m.group(2), m.group(3)
    else:
        parts = rest.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f'ARFF_BAD_ATTRIBUTE: {line[:150]}')
        name, kind = parts
    kind = kind.strip().lower().replace(' ', '')
    if kind not in {'numeric', 'real', 'integer', '{0,1}'}:
        raise ValueError(f'ARFF_UNSUPPORTED_ATTRIBUTE_TYPE: {name} {kind}')
    return name, kind


def parse_arff(data: bytes, labels: list[str]):
    """Strict numeric/binary dense or sparse ARFF, labels ordered by XML.

    Sparse omission means numeric zero or first nominal value (only {0,1} allowed).
    Missing values, repeated sparse indices, unknown features, and nonbinary labels abort.
    Returns scipy CSR X, uint8 Y, and the complete ordered attribute specification.
    """
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('ARFF_NOT_UTF8') from exc
    lines = iter(enumerate(text.splitlines(), 1))
    attributes = []
    found_data = False
    for lineno, raw in lines:
        line = raw.strip()
        if not line or line.startswith('%'):
            continue
        if re.match(r'^@attribute\s', line, re.I):
            attributes.append(_attr_name_type(line))
        elif re.match(r'^@data(?:\s|$)', line, re.I):
            found_data = True
            break
        elif not re.match(r'^@relation\s', line, re.I):
            raise ValueError(f'ARFF_UNEXPECTED_HEADER_LINE: {lineno}')
    if not found_data or not attributes:
        raise ValueError('ARFF_MISSING_HEADER_OR_DATA')
    attr_names = [x[0] for x in attributes]
    if len(attr_names) != len(set(attr_names)):
        raise ValueError('ARFF_DUPLICATE_ATTRIBUTE_NAMES')
    label_set = set(labels)
    if not label_set.issubset(attr_names):
        raise ValueError('XML_LABEL_MISSING_FROM_ARFF')
    label_positions = {attr_names.index(name): j for j, name in enumerate(labels)}
    feature_positions = {i: j for j, i in enumerate(
        i for i, name in enumerate(attr_names) if name not in label_set)}
    values, indices, indptr, ys = [], [], [0], []
    for lineno, raw in lines:
        line = raw.strip()
        if not line or line.startswith('%'):
            continue
        pairs = []
        if line.startswith('{'):
            if not line.endswith('}'):
                raise ValueError(f'ARFF_BAD_SPARSE_ROW: {lineno}')
            row = line[1:-1].strip()
            prior = -1
            if row:
                for token in row.split(','):
                    fields = token.strip().split(None, 1)
                    if len(fields) != 2:
                        raise ValueError(f'ARFF_BAD_SPARSE_ITEM: {lineno}')
                    pos, value = int(fields[0]), fields[1].strip()
                    if not 0 <= pos < len(attributes) or pos <= prior:
                        raise ValueError(f'ARFF_UNSORTED_DUPLICATE_OR_BAD_INDEX: {lineno}')
                    pairs.append((pos, value))
                    prior = pos
        else:
            row = next(csv.reader([line], skipinitialspace=True))
            if len(row) != len(attributes):
                raise ValueError(f'ARFF_DENSE_COLUMN_COUNT: {lineno}')
            pairs = enumerate(row)
        y = np.zeros(len(labels), dtype=np.uint8)
        for pos, value_text in pairs:
            if value_text.strip().strip('"\'') == '?':
                raise ValueError(f'ARFF_MISSING_VALUE: {lineno}')
            value = float(value_text.strip().strip('"\''))
            if not np.isfinite(value):
                raise ValueError(f'ARFF_NONFINITE: {lineno}')
            if pos in label_positions:
                if value not in (0.0, 1.0):
                    raise ValueError(f'ARFF_NONBINARY_LABEL: {lineno}')
                y[label_positions[pos]] = int(value)
            else:
                if attributes[pos][1] == '{0,1}' and value not in (0.0, 1.0):
                    raise ValueError(f'ARFF_NONBINARY_NOMINAL: {lineno}')
                if value:
                    values.append(value)
                    indices.append(feature_positions[pos])
        indptr.append(len(indices))
        ys.append(y)
    X = sparse.csr_matrix((np.asarray(values, dtype=np.float64),
                           np.asarray(indices, dtype=np.int32),
                           np.asarray(indptr, dtype=np.int64)),
                          shape=(len(ys), len(feature_positions)))
    X.sort_indices()
    Y = np.asarray(ys, dtype=np.uint8).reshape(len(ys), len(labels))
    return X, Y, attributes


def _load_mulan_members(members: dict[str, bytes], dataset: str):
    """Read official train/test members without shuffling or role assignment.

    Returns dictionary keys X_train, Y_train, X_test, Y_test, metadata.
    Only official train/test arrays are parsed; a union member is not parsed.
    """
    if dataset not in EXPECTED:
        raise ValueError(f'UNDECLARED_DATASET: {dataset}')
    xmls = [n for n in members if n.lower().endswith('.xml')]
    if len(xmls) != 1:
        raise ValueError(f'EXPECTED_ONE_XML: {xmls}')
    labels = xml_labels(members[xmls[0]])
    selected = {}
    for split in ['train', 'test']:
        found = [name for name in members if name.lower().endswith('.arff') and
                 re.search(rf'(?:^|[^a-z]){split}(?:[^a-z]|$)',
                           pathlib.PurePosixPath(name).stem.lower())]
        if len(found) != 1:
            raise ValueError(f'EXPECTED_ONE_OFFICIAL_{split.upper()}_MEMBER: {found}')
        selected[split] = found[0]
    result = {}
    attr_ref = None
    for split, name in selected.items():
        X, Y, attrs = parse_arff(members[name], labels)
        if attr_ref is None:
            attr_ref = attrs
        elif attrs != attr_ref:
            raise ValueError('TRAIN_TEST_ATTRIBUTE_MISMATCH')
        if X.shape[1] != EXPECTED[dataset]['n_features'] or Y.shape[1] != EXPECTED[dataset]['n_labels']:
            raise ValueError(f'DATASET_DIMENSION_MISMATCH: {split}, {X.shape}, {Y.shape}')
        if 'n_' + split in EXPECTED[dataset] and X.shape[0] != EXPECTED[dataset]['n_' + split]:
            raise ValueError(f'OFFICIAL_SPLIT_COUNT_MISMATCH: {split}, {X.shape[0]}')
        result['X_' + split], result['Y_' + split] = X, Y
    if result['X_train'].shape[0] + result['X_test'].shape[0] != EXPECTED[dataset]['n_total']:
        raise ValueError('DATASET_TOTAL_COUNT_MISMATCH')
    result['metadata'] = {
        'dataset': dataset,
        'selected_members': selected,
        'xml_member': xmls[0],
        'all_text_member_names': sorted(members),
        'member_sha256': {n: hashlib.sha256(members[n]).hexdigest()
                          for n in [xmls[0], *selected.values()]},
        'n_train': int(result['X_train'].shape[0]),
        'n_test': int(result['X_test'].shape[0]),
        'n_features': int(result['X_train'].shape[1]),
        'n_labels': len(labels),
        'attribute_names': [x[0] for x in attr_ref],
        'group_ids': None,
        'group_ids_note': 'Adapter preserves source row order; ARFF/XML group metadata not assumed.',
    }
    return result


def load_mulan_archive(path: str | pathlib.Path, dataset: str):
    """Read a local archive with libarchive; never downloads data."""
    path = pathlib.Path(path)
    result = _load_mulan_members(_archive_text_members(path), dataset)
    result['metadata']['raw_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['metadata']['raw_size_bytes'] = path.stat().st_size
    return result


def read_dataset(extracted_root: str | pathlib.Path, dataset: str):
    """Parent API: return X_train, y_train, X_test, y_test, metadata.

    Only XML and official train/test ARFF content is read. Other members are listed
    by relative name but their content is not opened. Symlinks are not accepted.
    """
    root = pathlib.Path(extracted_root).resolve()
    if not root.is_dir():
        raise ValueError('EXTRACTED_ROOT_NOT_DIRECTORY')
    members = {}
    all_names = []
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() not in {'.arff', '.xml'}:
            continue
        if p.is_symlink() or not p.is_file() or root not in p.resolve().parents:
            raise ValueError('NONREGULAR_EXTRACTED_MEMBER')
        name = p.relative_to(root).as_posix()
        all_names.append(name)
        if p.suffix.lower() == '.xml' or re.search(
                r'(?:^|[^a-z])(?:train|test)(?:[^a-z]|$)', p.stem.lower()):
            if p.stat().st_size > 512 * 1024 * 1024:
                raise ValueError('EXTRACTED_MEMBER_SIZE_LIMIT')
            members[name] = p.read_bytes()
    result = _load_mulan_members(members, dataset)
    result['metadata']['all_text_member_names'] = all_names
    return (result['X_train'], result['Y_train'], result['X_test'],
            result['Y_test'], result['metadata'])


def synthetic_self_test():
    """Tiny invented rows test parsing only; no benchmark data or model result."""
    labels = xml_labels(b'<labels xmlns="urn:test"><label name="l2"/><label name="l1"/></labels>')
    header = '@relation synthetic\n@attribute x1 numeric\n@attribute x2 {0,1}\n@attribute l1 {0,1}\n@attribute l2 {0,1}\n@data\n'
    dense = (header + '2,0,1,0\n0,1,0,1\n').encode()
    sparse_data = (header + '{0 2,2 1}\n{1 1,3 1}\n').encode()
    a, ya, aa = parse_arff(dense, labels)
    b, yb, ab = parse_arff(sparse_data, labels)
    assert (a != b).nnz == 0 and np.array_equal(ya, yb) and aa == ab
    assert np.array_equal(ya, [[0, 1], [1, 0]])
    for tail in ['{0 1,0 2}\n', '0,?,0,1\n', '0,0,2,1\n']:
        try:
            parse_arff((header + tail).encode(), labels)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed synthetic input was accepted')
    return {'synthetic_checks': 5, 'passed': True}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-test', action='store_true')
    args = parser.parse_args()
    if args.synthetic_test:
        print(json.dumps(synthetic_self_test(), sort_keys=True))
    else:
        parser.error('Use as an import after protocol freeze, or run --synthetic-test.')
