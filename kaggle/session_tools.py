"""Resolve a notebook version's real run ID, read live logs, or cancel that run.
The CLI's version number is not a run ID. The authenticated output redirect
identifies the /kf/<run-id>/ target even before outputs have been committed.
Never print the signed redirect URL or authentication headers.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiCancelKernelSessionRequest, ApiDownloadKernelOutputRequest,
    ApiGetKernelRequest, ApiGetKernelSessionLogsStreamRequest,
)

DEFAULT_KERNEL = 'rizwanabdullah05/secap-reproduction-t4x2'


def resolve(client, kernel, version=None):
    owner, slug = kernel.split('/')
    if version is None:
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        version = client.kernels.kernels_api_client.get_kernel(request).metadata.current_version_number
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug, request.kernel_slug = owner, slug
    request.version_number = int(version)
    request.file_path = '__notebook__.ipynb'
    url = ''
    try:
        response = client.kernels.kernels_api_client.download_kernel_output(request)
        url = response.url
    except Exception as error:
        response = getattr(error, 'response', None)
        if response is not None:
            url = response.url
    parsed = urlsplit(url)
    match = re.match(r'^/kf/(\d+)/', parsed.path)
    if parsed.hostname != 'www.kaggleusercontent.com' or not match:
        raise RuntimeError(f'Cannot resolve an authenticated run ID for version {version}')
    return int(version), int(match.group(1))


def observe(api, kernel=DEFAULT_KERNEL, version=None, seconds=10):
    lines = deque(maxlen=4096)
    with api.build_kaggle_client() as client:
        http = client._http_client
        http._init_session()
        send = http._session.send
        def bounded_send(request, **kwargs):
            kwargs['timeout'] = (10, max(3, seconds))
            return send(request, **kwargs)
        http._session.send = bounded_send
        version, run_id = resolve(client, kernel, version)
        request = ApiGetKernelSessionLogsStreamRequest()
        request.kernel_session_id = run_id
        request.wait_for_logs_url_seconds = 1
        started = time.monotonic()
        response = None
        try:
            response = client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
            if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                events = response.json()
                if isinstance(events, list):
                    lines.extend(event.get('data', '') for event in events)
            else:
                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if line and line.startswith('data:'):
                        try:
                            event = json.loads(line[5:])
                            if isinstance(event, dict):
                                lines.append(event.get('data', ''))
                        except ValueError:
                            pass
                    if time.monotonic() - started >= seconds:
                        break
        except Exception:
            if not lines:
                raise RuntimeError(f'Live logs unavailable for version {version}, run {run_id}') from None
        finally:
            if response is not None:
                response.close()
    return dict(version=version, run_id=run_id, logs=''.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['logs', 'cancel', 'resolve'])
    parser.add_argument('--kernel', default=DEFAULT_KERNEL)
    parser.add_argument('--version', type=int)
    parser.add_argument('--seconds', type=int, default=10)
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    if args.action == 'logs':
        result = observe(api, args.kernel, args.version, args.seconds)
        out = Path(__file__).parent / 'monitor'
        out.mkdir(exist_ok=True)
        (out / f'{args.kernel.split("/")[-1]}-version-{result["version"]}-live.log').write_text(result['logs'])
        print(f'Version {result["version"]}, run {result["run_id"]}')
        print(result['logs'][-8000:])
    else:
        with api.build_kaggle_client() as client:
            version, run_id = resolve(client, args.kernel, args.version)
            print(f'Verified authenticated redirect: version {version}, run {run_id}')
            if args.action == 'cancel':
                if args.version is None:
                    raise SystemExit('Cancellation requires an explicit --version')
                request = ApiCancelKernelSessionRequest()
                request.kernel_session_id = run_id
                print(client.kernels.kernels_api_client.cancel_kernel_session(request))
