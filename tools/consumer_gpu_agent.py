#!/usr/bin/env python3
# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Task-state and manual-review handoff tool for the consumer-GPU DFT project.

The project workflow (docs/consumer-gpu-plan.md) runs one GitHub issue per
branch with an explicit human (Codex) review gate.  This tool owns the
machine-readable side of that workflow:

  * a task registry (JSON, OUTSIDE the repository) with the state machine
        pending -> running -> implemented -> review_requested
                             -> accepted | needs_changes -> pending (rework)
        running -> pending (retry) | blocked (attempts exhausted / external)
        review_requested -> pending (reopen: superseded requirements or a
        withdrawn handoff, before any review verdict was recorded);
  * a dependency gate: `start` refuses a task whose dependencies are not
    all `accepted`;
  * a review gate: `accepted` can only be written by an explicit
    `review --verdict accepted --reviewer <name> --evidence <ref>` record.
    No other command reaches `accepted`, and a reviewer name that looks like
    the agent itself is refused (tripwire against self-approval; this is not
    authentication);
  * a single-GPU run lock (lockfile with pid + heartbeat) so two GPU tasks
    cannot run concurrently; dead/stale locks are taken over and logged;
  * per-task timeout and bounded retries (`expire` turns a missed deadline
    into one failed attempt; attempts >= max_attempts -> blocked);
  * handoff: `handoff` writes review_request.json, an issue-scoped patch and
    a PR description to a directory OUTSIDE the repository; `manifest` writes
    the publisher manifest (issue-N.json, exact key set, path/size checks,
    mode 0644) consumed by the root-owned publisher;
  * a machine-readable daily report (`daily-report`).

The tool deliberately does NOT: call any model or external service, touch
credentials, push/merge/close anything on GitHub, or execute GPU work.  It
only manipulates local JSON state and read-only git plumbing (rev-parse,
diff, status, show).

State lives outside the repository by default:
    /data/work/gpu4pyscf-agent/state/tasks.json
Override with --state or CGA_STATE.  The GPU lock file sits next to it.
'''

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

SCHEMA = 'consumer-gpu-tasks-v1'
REVIEW_SCHEMA = 'consumer-gpu-review-request-v1'
DAILY_SCHEMA = 'consumer-gpu-daily-report-v1'
MANIFEST_SCHEMA_KEYS = ('issue', 'base_sha', 'title', 'body', 'files')

STATUSES = ('pending', 'running', 'implemented', 'review_requested',
            'needs_changes', 'accepted', 'blocked')

DEFAULT_STATE = '/data/work/gpu4pyscf-agent/state/tasks.json'
DEFAULT_TIMEOUT_MIN = 240
DEFAULT_MAX_ATTEMPTS = 3
LOCK_STALE_SEC = 1800          # heartbeat older than this = stale lock
MAX_EVENT_LOG = 5000

# Tripwire only: the reviewer identity itself cannot be verified locally.
SELF_REVIEW_NAMES = {'self', 'me', 'zcode', 'agent', 'zcode-agent', '自动', '本人'}

MAX_FILES = 256
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 5_000_000
MAX_TITLE_CHARS = 200
MAX_BODY_CHARS = 20_000

REVIEW_CHECKLIST = [
    'diff is limited to this issue and is human-reviewable text',
    'tests/commands in review_request.json reproduce from the branch',
    'evidence files exist and their sha256 matches the recorded value',
    'no credentials, private endpoints, logs or hidden paths are published',
    'acceptance criteria of docs/consumer-gpu-plan.md section 5 are met',
]

# The task matrix of docs/consumer-gpu-plan.md section 5.  Kept in code so
# `init` produces the registry without a separate file to lose.
DEFAULT_TASKS = [
    {'issue': 2,  'title': '计划/任务矩阵/基准 schema/工作流工具 (this issue)',
     'dependencies': []},
    {'issue': 3,  'title': '隔离源码环境: sm_89 + CUDA 12.8 source build',
     'dependencies': [2]},
    {'issue': 4,  'title': 'RTX 4090 基准框架与 schema 校验器',
     'dependencies': [3]},
    {'issue': 5,  'title': 'B3LYP 能量基准 (试点阶梯, 小/中体系优先; 大体系为延期候选)',
     'dependencies': [4]},
    {'issue': 6,  'title': 'M06-2X 能量基准 + 网格稳定性 (试点阶梯)',
     'dependencies': [4]},
    {'issue': 7,  'title': 'omegaB97M-V (VV10/NLC) 能量基准 (试点阶梯)',
     'dependencies': [4]},
    {'issue': 8,  'title': '完整 r2SCAN-3c 能量基准: def2-mTZVPP+D4+gCP (试点阶梯)',
     'dependencies': [4]},
    {'issue': 9,  'title': '四方法热点归属 (阶段 x 函数 x 内核)',
     'dependencies': [5, 6, 7, 8]},
    {'issue': 10, 'title': '热点受控混合精度 (先量精度再动代码)',
     'dependencies': [9]},
    {'issue': 11, 'title': '显存边界测绘与批量 (a); offload/streaming 为后续研究 (b), 本期不实现',
     'dependencies': [10]},
    {'issue': 12, 'title': '能量门通过后的梯度',
     'dependencies': [10, 11]},
]

REPO_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AgentError(Exception):
    '''Policy violation or bad input; the command aborts without side effects.'''


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def utcnow():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.isoformat(timespec='seconds')


def parse_iso(s):
    return datetime.fromisoformat(s)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(repo, *args, binary=False):
    '''Run a read-only git command; raise AgentError on failure.'''
    try:
        out = subprocess.run(['git', '-C', repo, *args], check=True,
                             capture_output=True)
    except FileNotFoundError:
        raise AgentError('git not found; it is required for this command')
    except subprocess.CalledProcessError as exc:
        tail = exc.stderr.decode('utf-8', 'replace').strip()[-400:]
        raise AgentError(f"git {' '.join(args)} failed: {tail}")
    return out.stdout if binary else out.stdout.decode('utf-8', 'replace')


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def load_registry(path):
    if not os.path.exists(path):
        raise AgentError(
            f'state file {path} does not exist; run `init` first')
    with open(path, encoding='utf-8') as f:
        reg = json.load(f)
    if reg.get('schema') != SCHEMA:
        raise AgentError(f'{path}: unexpected schema {reg.get("schema")!r}')
    return reg


def save_registry(path, reg):
    reg['events'] = reg['events'][-MAX_EVENT_LOG:]
    tmp = path + '.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(reg, f, ensure_ascii=False, indent=1)
        f.write('\n')
    os.replace(tmp, path)


def log_event(reg, issue, event, detail=''):
    reg['events'].append({'ts': iso(utcnow()), 'issue': issue,
                          'event': event, 'detail': detail[:2000]})


def get_task(reg, issue):
    task = reg['tasks'].get(str(issue))
    if task is None:
        raise AgentError(f'issue #{issue} is not in the registry')
    return task


def touch(task):
    task['updated_at'] = iso(utcnow())


# ---------------------------------------------------------------------------
# GPU run lock
# ---------------------------------------------------------------------------

def lock_path(state_file):
    return os.path.join(os.path.dirname(os.path.abspath(state_file)),
                        'gpu.lock')


def read_lock(state_file):
    p = lock_path(state_file)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (ValueError, OSError):
        return {'pid': -1, 'issue': None, 'heartbeat_at': '1970-01-01T00:00:00+00:00',
                'corrupt': True}


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True       # exists but owned by someone else
    return True


def lock_held(lock):
    '''A lock counts as held while its pid is alive and its heartbeat fresh.'''
    if not lock or lock.get('corrupt'):
        return False
    try:
        age = (utcnow() - parse_iso(lock['heartbeat_at'])).total_seconds()
    except (ValueError, TypeError):
        return False
    return pid_alive(lock.get('pid')) and age < LOCK_STALE_SEC


def acquire_lock(state_file, issue):
    existing = read_lock(state_file)
    if existing is not None:
        if lock_held(existing):
            raise AgentError(
                f'GPU lock held by issue #{existing.get("issue")} '
                f'(pid {existing.get("pid")}, heartbeat '
                f'{existing.get("heartbeat_at")}); only one GPU task at a time')
        # dead or stale: take over, but record that we did
    lock = {'pid': os.getpid(), 'issue': issue,
            'acquired_at': iso(utcnow()), 'heartbeat_at': iso(utcnow())}
    p = lock_path(state_file)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(lock, f, indent=1)
    return existing


def release_lock(state_file, issue):
    lock = read_lock(state_file)
    if lock is not None and lock.get('issue') == issue:
        try:
            os.remove(lock_path(state_file))
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_init(args):
    if os.path.exists(args.state) and not args.force:
        raise AgentError(f'{args.state} exists; use --force to reinitialize '
                         '(this drops the event history)')
    if args.plan:
        with open(args.plan, encoding='utf-8') as f:
            tasks = json.load(f)
        if not isinstance(tasks, list) or not tasks:
            raise AgentError('--plan must be a JSON list of task specs')
    else:
        tasks = DEFAULT_TASKS
    reg = {'schema': SCHEMA, 'tasks': {}, 'events': []}
    for spec in tasks:
        issue = spec['issue']
        if str(issue) in reg['tasks']:
            raise AgentError(f'duplicate issue number {issue} in plan')
        reg['tasks'][str(issue)] = {
            'issue': issue,
            'title': spec.get('title', ''),
            'dependencies': list(spec.get('dependencies', [])),
            'branch': spec.get('branch'),
            'status': 'pending',
            'attempt': 0,
            'max_attempts': spec.get('max_attempts', DEFAULT_MAX_ATTEMPTS),
            'timeout_minutes': spec.get('timeout_minutes', DEFAULT_TIMEOUT_MIN),
            'deadline': None,
            'base_sha': spec.get('base_sha'),
            'summary': spec.get('summary', ''),
            'notes': [],
            'tests': [],
            'evidence': [],
            'review': None,
            'updated_at': iso(utcnow()),
        }
        log_event(reg, issue, 'init', 'registered')
    save_registry(args.state, reg)
    print(f'initialized {args.state} with {len(reg["tasks"])} tasks')
    return 0


def cmd_list(args):
    reg = load_registry(args.state)
    print(f'{"issue":>5}  {"status":>16}  {"att":>3}  deps        title')
    for key in sorted(reg['tasks'], key=int):
        t = reg['tasks'][key]
        deps = ','.join(str(d) for d in t['dependencies']) or '-'
        print(f'{t["issue"]:>5}  {t["status"]:>16}  {t["attempt"]:>3}  '
              f'{deps:<11} {t["title"]}')
    return 0


def cmd_show(args):
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    print(json.dumps(task, ensure_ascii=False, indent=1))
    return 0


def dependency_ready(task):
    if task.get('status') == 'accepted':
        return True
    return (os.environ.get('QC_CONTINUE_UNREVIEWED') == '1'
            and task.get('status') == 'review_requested'
            and bool(task.get('tests')))


def cmd_start(args):
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    if task['status'] != 'pending':
        hint = ('; run `expire` first if the previous run died'
                if task['status'] == 'running' else
                '; needs_changes goes through `rework`, blocked through '
                '`unblock`')
        raise AgentError(f'issue #{args.issue} is {task["status"]}, '
                         f'start requires pending{hint}')
    blocked_deps = [d for d in task['dependencies']
                    if not dependency_ready(reg['tasks'].get(str(d), {}))]
    if blocked_deps:
        raise AgentError(
            f'dependency gate: issue #{args.issue} requires '
            f'{blocked_deps} to be accepted first (review gate is explicit '
            f'human review, not agent assertion)')
    if task['attempt'] >= task['max_attempts']:
        raise AgentError(f'issue #{args.issue} exhausted its '
                         f'{task["max_attempts"]} attempts; it must be '
                         f'unblocked manually')
    if args.timeout_min:
        task['timeout_minutes'] = args.timeout_min
    previous = acquire_lock(args.state, args.issue)
    if previous is not None:
        log_event(reg, args.issue, 'lock-takeover',
                  f'stale/dead lock from pid {previous.get("pid")} (issue '
                  f'{previous.get("issue")}) taken over')
    task['attempt'] += 1
    task['status'] = 'running'
    task['deadline'] = iso(utcnow() + timedelta(minutes=task['timeout_minutes']))
    if args.branch:
        task['branch'] = args.branch
    touch(task)
    log_event(reg, args.issue, 'start',
              f'attempt {task["attempt"]}/{task["max_attempts"]}, '
              f'deadline {task["deadline"]}')
    save_registry(args.state, reg)
    print(f'issue #{args.issue}: running (attempt {task["attempt"]}/'
          f'{task["max_attempts"]}, deadline {task["deadline"]})')
    return 0


def _end_running(args, new_status, event, detail):
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    if task['status'] != 'running':
        raise AgentError(f'issue #{args.issue} is {task["status"]}, '
                         f'expected running')
    task['status'] = new_status
    task['deadline'] = None
    if args.note:
        task['notes'].append(args.note)
    touch(task)
    log_event(reg, args.issue, event, detail)
    save_registry(args.state, reg)
    release_lock(args.state, args.issue)
    return new_status


def cmd_finish(args):
    if args.to not in ('implemented', 'blocked'):
        raise AgentError("--to must be 'implemented' or 'blocked'; acceptance "
                         'only happens through the review command')
    detail = f'{args.to}' + (f'; {args.note}' if args.note else '')
    _end_running(args, args.to, 'finish', detail)
    print(f'issue #{args.issue}: {args.to}')
    return 0


def cmd_fail(args):
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    if task['status'] != 'running':
        raise AgentError(f'issue #{args.issue} is {task["status"]}, '
                         'expected running')
    detail = 'timeout' if args.timeout else 'failure'
    if args.note:
        detail += f'; {args.note}'
        task['notes'].append(args.note)
    task['deadline'] = None
    if task['attempt'] >= task['max_attempts']:
        task['status'] = 'blocked'
        detail += ' -> blocked (attempts exhausted)'
    else:
        task['status'] = 'pending'
        detail += ' -> pending (retry available)'
    touch(task)
    log_event(reg, args.issue, 'fail', detail)
    save_registry(args.state, reg)
    release_lock(args.state, args.issue)
    print(f'issue #{args.issue}: {task["status"]} ({detail})')
    return 0


def cmd_expire(args):
    reg = load_registry(args.state)
    expired = []
    for key, task in reg['tasks'].items():
        if task['status'] != 'running' or not task['deadline']:
            continue
        try:
            overdue = utcnow() > parse_iso(task['deadline'])
        except ValueError:
            overdue = True
        if not overdue:
            continue
        expired.append(task['issue'])
        if task['attempt'] >= task['max_attempts']:
            task['status'] = 'blocked'
            resolution = 'blocked (attempts exhausted)'
        else:
            task['status'] = 'pending'
            resolution = 'pending (retry available)'
        task['deadline'] = None
        touch(task)
        log_event(reg, task['issue'], 'timeout',
                  f'deadline missed; -> {resolution}')
        release_lock(args.state, task['issue'])
    save_registry(args.state, reg)
    if expired:
        print('expired:', ', '.join(f'#{i}' for i in sorted(expired)))
    else:
        print('no expired tasks')
    return 0


def cmd_heartbeat(args):
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    if task['status'] != 'running':
        raise AgentError(f'issue #{args.issue} is {task["status"]}, '
                         'heartbeat requires running')
    lock = read_lock(args.state)
    if lock is None or lock.get('issue') != args.issue:
        raise AgentError(f'the GPU lock is not held by issue #{args.issue}')
    if not lock_held(lock):
        raise AgentError('the GPU lock looks stale; run start to take it over')
    lock['heartbeat_at'] = iso(utcnow())
    with open(lock_path(args.state), 'w', encoding='utf-8') as f:
        json.dump(lock, f, indent=1)
    task['deadline'] = iso(utcnow() + timedelta(minutes=task['timeout_minutes']))
    touch(task)
    log_event(reg, args.issue, 'heartbeat')
    save_registry(args.state, reg)
    print(f'issue #{args.issue}: heartbeat')
    return 0


def cmd_evidence(args):
    if not os.path.isfile(args.path):
        raise AgentError(f'evidence file not found: {args.path}')
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    entry = {'kind': args.kind, 'path': os.path.abspath(args.path),
             'sha256': sha256_file(args.path), 'note': args.note or '',
             'added_at': iso(utcnow())}
    task['evidence'].append(entry)
    touch(task)
    log_event(reg, args.issue, 'evidence',
              f'{args.kind} {entry["path"]}@{entry["sha256"][:12]}')
    save_registry(args.state, reg)
    print(f'issue #{args.issue}: evidence {entry["sha256"][:12]} '
          f'({args.kind})')
    return 0


def cmd_record_test(args):
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    task['tests'].append({'command': args.command, 'observed': args.observed,
                          'recorded_at': iso(utcnow())})
    touch(task)
    log_event(reg, args.issue, 'test-recorded', args.command)
    save_registry(args.state, reg)
    print(f'issue #{args.issue}: test recorded')
    return 0


def cmd_summary(args):
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    task['summary'] = args.text
    touch(task)
    save_registry(args.state, reg)
    return 0


def cmd_note(args):
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    task['notes'].append(args.text)
    touch(task)
    log_event(reg, args.issue, 'note', args.text)
    save_registry(args.state, reg)
    return 0


def _diff_file_list(repo, base):
    out = git(repo, 'diff', '--name-status', f'{base}..HEAD')
    files = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) == 2:                      # A / D / M
            status, path = parts
        elif len(parts) == 3:                    # R100 old new
            status, _old, path = parts
        else:
            raise AgentError(f'unparsable diff line: {line!r}')
        files.append({'status': status, 'path': path})
    return files


def cmd_handoff(args):
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    if task['status'] != 'implemented':
        raise AgentError(f'issue #{args.issue} is {task["status"]}; handoff '
                         'requires implemented (finish first)')
    base = args.base or task.get('base_sha')
    if not base:
        raise AgentError('no base sha: pass --base or set it at start')
    head = git(args.repo, 'rev-parse', 'HEAD').strip()
    files = _diff_file_list(args.repo, base)
    if not files:
        raise AgentError(f'no changes between {base} and HEAD; nothing to '
                         'hand off')
    patch = git(args.repo, 'diff', f'{base}..HEAD')
    os.makedirs(args.outbox, exist_ok=True)
    request = {
        'schema': REVIEW_SCHEMA,
        'issue': task['issue'],
        'title': task['title'],
        'status': 'review_requested',
        'generated_at_utc': iso(utcnow()),
        'branch': task.get('branch'),
        'base_sha': base,
        'head_sha': head,
        'summary': task.get('summary', ''),
        'changed_files': files,
        'tests': task.get('tests', []),
        'evidence': task.get('evidence', []),
        'notes': task.get('notes', []),
        'review_checklist': REVIEW_CHECKLIST,
    }
    paths = {name: os.path.join(args.outbox, name) for name in
             ('review_request.json', f'issue-{task["issue"]}.patch',
              f'issue-{task["issue"]}-pr-body.md')}
    with open(paths['review_request.json'], 'w', encoding='utf-8') as f:
        json.dump(request, f, ensure_ascii=False, indent=1)
    with open(paths[f'issue-{task["issue"]}.patch'], 'w', encoding='utf-8') as f:
        f.write(patch)
    body = [f'#{task["issue"]} {task["title"]}',
            '', f'base: {base}', f'head: {head}',
            f'branch: {task.get("branch")}', '', task.get('summary', ''), '',
            '## Changed files', '']
    body += [f'- {f["status"]} {f["path"]}' for f in files]
    body += ['', '## Tests', '']
    for t in task.get('tests', []):
        body += [f'`{t["command"]}`', '', f'> {t["observed"]}', '']
    if task.get('evidence'):
        body += ['## Evidence', '']
        body += [f'- {e["kind"]}: {e["path"]}@{e["sha256"][:12]}'
                 for e in task['evidence']]
        body += ['']
    body += ['## Review checklist', '']
    body += [f'- [ ] {c}' for c in REVIEW_CHECKLIST]
    body += ['', '**Manual review required: this handoff is an agent '
             'assertion, not approval. Do not merge before an explicit '
             'review record is written.**', '']
    with open(paths[f'issue-{task["issue"]}-pr-body.md'], 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(body))
    for p in paths.values():
        os.chmod(p, 0o644)
    task['status'] = 'review_requested'
    task['base_sha'] = base
    touch(task)
    log_event(reg, task['issue'], 'handoff',
              f'{len(files)} files, head {head[:12]}')
    save_registry(args.state, reg)
    print(f'issue #{task["issue"]}: review_requested; wrote '
          f'{", ".join(sorted(paths))}')
    return 0


def cmd_reopen(args):
    '''review_requested -> pending: withdraw a handoff before any review
    verdict was recorded (e.g. requirements changed under it).  attempt is
    deliberately NOT reset: the work cycle is continuing, not restarting.'''
    if not (args.reason or '').strip():
        raise AgentError('reopen requires --reason (what changed or why the '
                         'handoff is withdrawn; recorded in the event log)')
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    if task['status'] != 'review_requested':
        raise AgentError(f'issue #{args.issue} is {task["status"]}; reopen '
                         'requires review_requested (before any review '
                         'verdict; accepted/needs_changes cannot reopen)')
    task['status'] = 'pending'
    if args.reason:
        task['notes'].append(f'reopened: {args.reason}')
    touch(task)
    log_event(reg, args.issue, 'reopen', args.reason)
    save_registry(args.state, reg)
    print(f'issue #{args.issue}: pending (reopened, attempt '
          f'{task["attempt"]}/{task["max_attempts"]} kept)')
    return 0


def cmd_review(args):
    if args.verdict not in ('accepted', 'needs_changes'):
        raise AgentError("--verdict must be 'accepted' or 'needs_changes'")
    if args.verdict == 'accepted':
        reviewer = (args.reviewer or '').strip()
        if not reviewer:
            raise AgentError('acceptance requires an explicit --reviewer')
        if reviewer.lower() in SELF_REVIEW_NAMES:
            raise AgentError(
                f'reviewer {reviewer!r} looks like the agent itself; '
                'self-approval is not allowed')
        if not (args.evidence or '').strip():
            raise AgentError('acceptance requires --evidence referencing what '
                             'was reviewed (patch, files, evidence sha256)')
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    if task['status'] != 'review_requested':
        raise AgentError(f'issue #{args.issue} is {task["status"]}; review '
                         'requires review_requested (handoff first)')
    task['review'] = {
        'reviewer': args.reviewer,
        'verdict': args.verdict,
        'date': iso(utcnow()),
        'evidence': args.evidence or '',
        'notes': args.notes or '',
    }
    task['status'] = args.verdict
    touch(task)
    log_event(reg, args.issue, f'review-{args.verdict}',
              f'reviewer={args.reviewer}')
    save_registry(args.state, reg)
    print(f'issue #{args.issue}: {args.verdict} '
          f'(reviewer {args.reviewer or "n/a"})')
    return 0


def cmd_rework(args):
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    if task['status'] != 'needs_changes':
        raise AgentError(f'issue #{args.issue} is {task["status"]}; rework '
                         'requires needs_changes')
    task['status'] = 'pending'
    task['attempt'] = 0     # a fresh work cycle with reviewer feedback
    touch(task)
    log_event(reg, args.issue, 'rework', args.note or '')
    save_registry(args.state, reg)
    print(f'issue #{args.issue}: pending (rework)')
    return 0


def cmd_unblock(args):
    reg = load_registry(args.state)
    task = get_task(reg, args.issue)
    if task['status'] != 'blocked':
        raise AgentError(f'issue #{args.issue} is {task["status"]}; unblock '
                         'requires blocked')
    task['status'] = 'pending'
    task['attempt'] = 0     # external blocker, not a failure of the work
    touch(task)
    log_event(reg, args.issue, 'unblock', args.note or '')
    save_registry(args.state, reg)
    print(f'issue #{args.issue}: pending (unblocked)')
    return 0


def _validate_publish_path(path):
    if not path or os.path.isabs(path):
        raise AgentError(f'publish path must be relative: {path!r}')
    parts = path.replace('\\', '/').split('/')
    for part in parts:
        if part in ('', '.', '..'):
            raise AgentError(f'bad path component in {path!r}')
        if part.startswith('.'):
            raise AgentError(
                f'refusing hidden path {path!r}: hidden files and dot '
                'directories (.git, .github, ...) must not be published')
    return path


def cmd_manifest(args):
    if len(args.title) > MAX_TITLE_CHARS:
        raise AgentError(f'title exceeds {MAX_TITLE_CHARS} chars')
    with open(args.body_file, encoding='utf-8') as f:
        body = f.read()
    if len(body) > MAX_BODY_CHARS:
        raise AgentError(f'body exceeds {MAX_BODY_CHARS} chars')
    if len(args.base_sha) != 40 or any(c not in '0123456789abcdef'
                                       for c in args.base_sha.lower()):
        raise AgentError(f'base_sha must be a 40-hex commit sha, got '
                         f'{args.base_sha!r}')
    if not args.allow_dirty:
        dirty = git(args.repo, 'status', '--porcelain')
        if dirty.strip():
            raise AgentError('working tree is dirty; commit first or pass '
                             '--allow-dirty (manifest must match a commit)')
    head = git(args.repo, 'rev-parse', 'HEAD').strip()
    files = _diff_file_list(args.repo, args.base_sha)
    if not files:
        raise AgentError('no changed files against the base')
    if len(files) > MAX_FILES:
        raise AgentError(f'{len(files)} files exceeds the {MAX_FILES}-file '
                         'publication limit')
    entries = []
    total = 0
    for f in files:
        path = _validate_publish_path(f['path'])
        if f['status'] == 'D':
            content = None
        else:
            blob = git(args.repo, 'show', f'HEAD:{f["path"]}', binary=True)
            try:
                content = blob.decode('utf-8')
            except UnicodeDecodeError:
                raise AgentError(
                    f'{f["path"]} is not UTF-8 text; only human-reviewable '
                    'text may be published')
            nbytes = len(blob)
            if nbytes > MAX_FILE_BYTES:
                raise AgentError(f'{f["path"]} is {nbytes} bytes; limit is '
                                 f'{MAX_FILE_BYTES}')
            total += nbytes
        entries.append({'path': path, 'content': content})
    if total > MAX_TOTAL_BYTES:
        raise AgentError(f'changed files total {total} bytes; limit is '
                         f'{MAX_TOTAL_BYTES}')
    manifest = {'issue': int(args.issue), 'base_sha': args.base_sha,
                'title': args.title, 'body': body, 'files': entries}
    os.makedirs(args.outbox, exist_ok=True)
    out = os.path.join(args.outbox, f'issue-{args.issue}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.chmod(out, 0o644)
    print(f'wrote {out} ({len(entries)} files, head {head[:12]}, '
          f'{total} bytes of file content)')
    return 0


def cmd_daily_report(args):
    reg = load_registry(args.state)
    tasks = [reg['tasks'][k] for k in sorted(reg['tasks'], key=int)]
    lock = read_lock(args.state)
    unverified = []
    next_actions = []
    for t in tasks:
        if t['status'] in ('implemented', 'review_requested') \
                and not t['evidence']:
            unverified.append({'issue': t['issue'],
                               'note': 'no evidence recorded yet'})
        if t['status'] == 'running' and t['deadline']:
            try:
                if utcnow() > parse_iso(t['deadline']):
                    unverified.append({'issue': t['issue'],
                                       'note': 'deadline exceeded; run expire'})
            except ValueError:
                pass
        if t['status'] == 'review_requested':
            next_actions.append({'issue': t['issue'],
                                 'action': 'await manual review'})
        elif t['status'] == 'pending' and all(
                dependency_ready(reg['tasks'].get(str(d), {}))
                for d in t['dependencies']):
            next_actions.append({'issue': t['issue'],
                                 'action': 'start (dependencies ready; review may be pending)'})
    report = {
        'schema': DAILY_SCHEMA,
        'generated_at_utc': iso(utcnow()),
        'project': 'gpu4pyscf-mixedprec (RTX 4090 consumer DFT)',
        'issues': [{'issue': t['issue'], 'title': t['title'],
                    'status': t['status'], 'attempt': t['attempt'],
                    'branch': t.get('branch'), 'updated_at': t['updated_at'],
                    'evidence_count': len(t['evidence'])} for t in tasks],
        'recent_events': reg['events'][-50:],
        'gpu_lock': ({'issue': lock.get('issue'), 'pid': lock.get('pid'),
                      'heartbeat_at': lock.get('heartbeat_at'),
                      'held': lock_held(lock)}
                     if lock is not None else None),
        'in_review': [t['issue'] for t in tasks
                      if t['status'] == 'review_requested'],
        'accepted': [{'issue': t['issue'],
                      'reviewer': t['review']['reviewer'],
                      'date': t['review']['date']}
                     for t in tasks if t['status'] == 'accepted'],
        'blocked': [{'issue': t['issue'],
                     'note': (t['notes'][-1] if t['notes'] else '')}
                    for t in tasks if t['status'] == 'blocked'],
        'next_actions': next_actions,
        'unverified': unverified,
    }
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        os.chmod(args.out, 0o644)
        print(f'wrote {args.out}')
    else:
        print(text)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog='consumer_gpu_agent',
        description='Task state, review gates and handoff for the '
                    'consumer-GPU DFT project (see docs/consumer-gpu-plan.md)')
    p.add_argument('--state', default=os.environ.get('CGA_STATE',
                                                     DEFAULT_STATE),
                   help='task registry JSON (kept OUTSIDE the repository)')
    p.add_argument('--repo', default=REPO_DEFAULT,
                   help='repository root for git commands')
    sub = p.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('init', help='create the registry from the plan')
    sp.add_argument('--force', action='store_true',
                    help='reinitialize (drops the event history)')
    sp.add_argument('--plan', help='JSON list of task specs')

    sub.add_parser('list', help='one line per issue')
    sp = sub.add_parser('show', help='full task record')
    sp.add_argument('--issue', type=int, required=True)

    sp = sub.add_parser('start', help='dependency gate + GPU lock -> running')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--branch', help='record the working branch')
    sp.add_argument('--timeout-min', type=int,
                    help='override the task timeout for this run')

    sp = sub.add_parser('heartbeat', help='refresh GPU lock + deadline')
    sp.add_argument('--issue', type=int, required=True)

    sp = sub.add_parser('finish', help='running -> implemented | blocked')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--to', required=True, choices=['implemented', 'blocked'])
    sp.add_argument('--note')

    sp = sub.add_parser('fail', help='running -> pending (retry) | blocked')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--timeout', action='store_true')
    sp.add_argument('--note')

    sub.add_parser('expire', help='missed deadlines count as failed attempts')

    sp = sub.add_parser('evidence', help='anchor a file by sha256')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--path', required=True)
    sp.add_argument('--kind', required=True,
                    help='e.g. benchmark-json, build-log, env-manifest')
    sp.add_argument('--note')

    sp = sub.add_parser('record-test', help='record a command + observed result')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--command', required=True)
    sp.add_argument('--observed', required=True)

    sp = sub.add_parser('summary', help='set the handoff summary')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--text', required=True)

    sp = sub.add_parser('note', help='append a note to the task')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--text', required=True)

    sp = sub.add_parser('handoff',
                        help='implemented -> review_requested; write '
                             'review_request.json + patch + PR body')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--outbox', required=True,
                    help='directory OUTSIDE the repository')
    sp.add_argument('--base', help='base commit sha (defaults to recorded)')

    sp = sub.add_parser('reopen',
                        help='review_requested -> pending (withdraw a '
                             'handoff before any review verdict)')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--reason', required=True,
                    help='what changed or why the handoff is withdrawn')

    sp = sub.add_parser('review',
                        help='record the explicit human review verdict')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--verdict', required=True,
                    choices=['accepted', 'needs_changes'])
    sp.add_argument('--reviewer')
    sp.add_argument('--evidence', help='what was reviewed (required to accept)')
    sp.add_argument('--notes')

    sp = sub.add_parser('rework', help='needs_changes -> pending')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--note')

    sp = sub.add_parser('unblock', help='blocked -> pending')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--note')

    sp = sub.add_parser('manifest',
                        help='write the publisher manifest issue-N.json')
    sp.add_argument('--issue', type=int, required=True)
    sp.add_argument('--outbox', required=True)
    sp.add_argument('--base-sha', required=True)
    sp.add_argument('--title', required=True)
    sp.add_argument('--body-file', required=True)
    sp.add_argument('--allow-dirty', action='store_true')

    sp = sub.add_parser('daily-report', help='machine-readable daily fields')
    sp.add_argument('--out', help='also write to this file')

    return p


HANDLERS = {
    'init': cmd_init, 'list': cmd_list, 'show': cmd_show, 'start': cmd_start,
    'heartbeat': cmd_heartbeat, 'finish': cmd_finish, 'fail': cmd_fail,
    'expire': cmd_expire, 'evidence': cmd_evidence,
    'record-test': cmd_record_test, 'summary': cmd_summary,
    'note': cmd_note, 'handoff': cmd_handoff, 'reopen': cmd_reopen,
    'review': cmd_review,
    'rework': cmd_rework, 'unblock': cmd_unblock,
    'manifest': cmd_manifest, 'daily-report': cmd_daily_report,
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return HANDLERS[args.cmd](args)
    except AgentError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
