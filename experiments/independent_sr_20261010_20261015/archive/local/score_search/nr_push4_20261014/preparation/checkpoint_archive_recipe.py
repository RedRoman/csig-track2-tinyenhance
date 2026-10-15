"""Lossless byte recipe for a raw torch ZIP whose tensor bytes are in its resume.

No torch import or unpickling. Store raw ZIP gaps verbatim and refer to verified
resume ZIP entries for tensor payloads. A restored raw file must match its exact
original SHA256, including ZIP headers, pickle metadata and serialization ID.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import zipfile


def need(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def entry_digest(archive, info):
    h = hashlib.sha256()
    with archive.open(info) as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def data_start(stream, info):
    stream.seek(info.header_offset)
    header = stream.read(30)
    need(header[:4] == b'PK\x03\x04', 'Invalid local ZIP header')
    return info.header_offset + 30 + int.from_bytes(header[26:28], 'little') + int.from_bytes(header[28:30], 'little')


def create(raw, resume, directory):
    raw, resume, directory = Path(raw).resolve(), Path(resume).resolve(), Path(directory).resolve()
    need(raw != resume and not directory.exists(), 'Use distinct source files and a new recipe directory')
    directory.mkdir(parents=True)
    tokens = {p: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) for p in (raw, resume)}
    skeleton = directory / 'raw_zip_gaps.bin'
    segments, gap_position = [], 0
    with zipfile.ZipFile(raw) as rz, zipfile.ZipFile(resume) as sz, raw.open('rb') as source, skeleton.open('wb') as gaps:
        raw_entries = [i for i in rz.infolist() if re.fullmatch(r'.+/data/\d+', i.filename)]
        need(len(raw_entries) == 514, 'Expected exactly 514 LoRA tensor ZIP entries')
        sizes = {i.file_size for i in raw_entries}
        resume_map = {}
        for info in sz.infolist():
            if re.fullmatch(r'.+/data/\d+', info.filename) and info.file_size in sizes:
                key = (info.file_size, entry_digest(sz, info))
                resume_map.setdefault(key, info.filename)
        ranges = []
        for info in raw_entries:
            need(info.compress_type == zipfile.ZIP_STORED, 'Raw tensor entries must be uncompressed')
            sha = entry_digest(rz, info)
            key = (info.file_size, sha)
            need(key in resume_map, 'Raw tensor bytes absent from matching resume')
            ranges.append((data_start(source, info), info.file_size, sha, resume_map[key]))
        position = 0
        for offset, length, sha, entry in sorted(ranges):
            need(offset >= position, 'Overlapping raw tensor ZIP payloads')
            if offset > position:
                source.seek(position)
                data = source.read(offset - position)
                need(len(data) == offset - position, 'Incomplete raw gap read')
                gaps.write(data)
                segments.append({'kind': 'gap', 'offset': gap_position, 'bytes': len(data)})
                gap_position += len(data)
            segments.append({'kind': 'resume_entry', 'name': entry, 'bytes': length, 'sha256': sha})
            position = offset + length
        source.seek(position)
        tail = source.read()
        gaps.write(tail)
        segments.append({'kind': 'gap', 'offset': gap_position, 'bytes': len(tail)})
    for p, token in tokens.items():
        need((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) == token, 'Checkpoint changed while creating recipe')
    recipe = {'schema': 'raw_torch_zip_exact_byte_recipe_v1', 'raw_name': raw.name,
              'raw_bytes': raw.stat().st_size, 'raw_sha256': digest(raw),
              'resume_name': resume.name, 'resume_bytes': resume.stat().st_size, 'resume_sha256': digest(resume),
              'gaps_file': skeleton.name, 'gaps_sha256': digest(skeleton),
              'tensor_entries': 514, 'segments': segments,
              'interpretation': 'Raw tensor values and exact original raw ZIP bytes can both be reconstructed from the retained exact resume and these raw ZIP gaps.'}
    (directory / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    print(json.dumps({k: v for k, v in recipe.items() if k != 'segments'}))


def restore(directory, resume, output):
    directory, resume, output = Path(directory).resolve(), Path(resume).resolve(), Path(output).resolve()
    recipe = json.loads((directory / 'recipe.json').read_text())
    need(recipe['schema'] == 'raw_torch_zip_exact_byte_recipe_v1', 'Unknown recipe schema')
    need(not output.exists() and output != resume, 'Restore requires a new output path')
    need(resume.stat().st_size == recipe['resume_bytes'] and digest(resume) == recipe['resume_sha256'], 'Resume identity differs')
    gap = directory / recipe['gaps_file']
    need(gap.parent == directory and digest(gap) == recipe['gaps_sha256'], 'Raw ZIP gaps differ')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(resume) as archive, gap.open('rb') as gaps, output.open('xb') as stream:
        for segment in recipe['segments']:
            if segment['kind'] == 'gap':
                gaps.seek(segment['offset'])
                data = gaps.read(segment['bytes'])
                need(len(data) == segment['bytes'], 'Incomplete gap payload')
                stream.write(data)
            else:
                need(segment['kind'] == 'resume_entry', 'Unknown segment kind')
                info = archive.getinfo(segment['name'])
                need(info.file_size == segment['bytes'], 'Resume tensor entry length differs')
                h, total = hashlib.sha256(), 0
                with archive.open(info) as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        h.update(block); total += len(block); stream.write(block)
                need(total == segment['bytes'] and h.hexdigest() == segment['sha256'], 'Resume tensor entry differs')
    need(output.stat().st_size == recipe['raw_bytes'] and digest(output) == recipe['raw_sha256'], 'Reconstructed raw ZIP differs')
    print(json.dumps({'status': 'PASS', 'restored_raw_sha256': recipe['raw_sha256'],
                      'bytes': recipe['raw_bytes'], 'tensor_entries': recipe['tensor_entries']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('create')
    c.add_argument('--raw', type=Path, required=True); c.add_argument('--resume', type=Path, required=True)
    c.add_argument('--directory', type=Path, required=True)
    r = commands.add_parser('restore')
    r.add_argument('--directory', type=Path, required=True); r.add_argument('--resume', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'create':create(args.raw, args.resume, args.directory)
    else:restore(args.directory, args.resume, args.output)
