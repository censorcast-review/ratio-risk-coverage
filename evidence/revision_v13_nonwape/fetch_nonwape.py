"""Fetch the two declared public Mulan archives after the local code freeze.

No dataset substitution. Transport failures are recorded as failures.
"""
import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from run_nonwape import HERE, assert_freeze, sha, utc, write_json


class OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = (urllib.parse.urlparse(newurl).hostname or '').lower()
        if not (host == 'sourceforge.net' or host.endswith('.sourceforge.net') or
                host == 'sf.net' or host.endswith('.sf.net')):
            raise RuntimeError('undeclared download redirect host: '+host)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(protocol_path):
    protocol = assert_freeze(protocol_path)
    root = HERE/'raw'
    root.mkdir(exist_ok=True)
    opener = urllib.request.build_opener(OfficialRedirects())
    summary = {}
    for dataset, spec in protocol['datasets'].items():
        receipt = HERE/'evidence'/('DOWNLOAD_'+dataset+'.json')
        target = root/(dataset+'.rar')
        temporary = target.with_suffix('.rar.partial')
        if receipt.exists() or target.exists():
            raise RuntimeError('download already recorded; no implicit restart')
        record = {'dataset': dataset, 'url': spec['url'], 'started_utc': utc(),
                  'protocol_sha256': sha(protocol_path), 'raw_inspected_before_freeze': False}
        write_json(receipt, record)
        try:
            request = urllib.request.Request(spec['url'], headers={'User-Agent': 'CENSORCAST-research-replication/13'})
            with opener.open(request, timeout=60) as response:
                record['http_status'] = response.status
                record['final_source_host'] = urllib.parse.urlparse(response.url).hostname
                record['content_type'] = response.headers.get('Content-Type')
                size = 0
                with open(temporary, 'wb') as output:
                    while True:
                        block = response.read(1024*1024)
                        if not block:
                            break
                        output.write(block)
                        size += len(block)
                        if size > 512*1024*1024:
                            raise RuntimeError('raw archive exceeds predeclared 512 MiB limit')
                    output.flush()
                    os.fsync(output.fileno())
            with open(temporary, 'rb') as stream:
                signature = stream.read(8)
            if not signature.startswith(b'Rar!\x1a\x07'):
                raise RuntimeError('download did not return a RAR archive')
            os.replace(temporary, target)
            record.update({'status': 'downloaded', 'bytes': target.stat().st_size,
                           'sha256': sha(target), 'completed_utc': utc()})
        except Exception as exc:
            record.update({'status': 'download_failed', 'exception': type(exc).__name__,
                           'message': str(exc), 'completed_utc': utc()})
        write_json(receipt, record)
        summary[dataset] = record
        print(dataset, record['status'], record.get('bytes', record.get('message')), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default=str(HERE/'evidence'/'PROTOCOL.json'))
    args = parser.parse_args()
    fetch(args.protocol)
