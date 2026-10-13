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

'''Behavioral tests for tools/consumer_gpu_agent.py.

These test the POLICY the tool enforces (dependency gate, review gate,
bounded retries, GPU lock exclusion, handoff artifacts, publisher manifest
validation), not its internal structure.  No GPU, no pyscf, no network: the
tool is deliberately pure stdlib and all state lives in temp directories.
'''

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS_DIR)

import consumer_gpu_agent as cga  # noqa: E402


def run(*argv):
    '''Invoke the CLI in-process; return (rc, stdout, stderr).

    SystemExit (argparse usage errors) is folded into rc=2, matching what a
    shell would see.
    '''
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cga.main([str(a) for a in argv])
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 2
    return rc, out.getvalue(), err.getvalue()


class Env:
    '''A temp workspace: task registry + a small git repository.'''

    def __init__(self, plan, repo_files=None, hidden_file=False):
        self.dir = tempfile.TemporaryDirectory()
        self.base = self.dir.name
        self.state = os.path.join(self.base, 'state', 'tasks.json')
        self.outbox = os.path.join(self.base, 'outbox')
        plan_path = os.path.join(self.base, 'plan.json')
        with open(plan_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f)
        rc, _, err = run('--state', self.state, 'init', '--plan', plan_path)
        assert rc == 0, err
        self.repo, self.base_sha = self._make_repo(repo_files or {},
                                                   hidden_file)

    def _make_repo(self, repo_files, hidden_file):
        repo = os.path.join(self.base, 'repo')
        os.makedirs(repo)

        def g(*a):
            subprocess.run(['git', '-C', repo, *a], check=True,
                           capture_output=True)

        g('init', '-q')
        g('config', 'user.email', 'test@example.com')
        g('config', 'user.name', 'test')
        with open(os.path.join(repo, 'base.txt'), 'w', encoding='utf-8') as f:
            f.write('base\n')
        g('add', '.')
        g('commit', '-q', '-m', 'base')
        base = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
        for name, content in repo_files.items():
            with open(os.path.join(repo, name), 'w', encoding='utf-8') as f:
                f.write(content)
        if hidden_file:
            with open(os.path.join(repo, '.hidden'), 'w',
                      encoding='utf-8') as f:
                f.write('must not be published\n')
        g('add', '-A')
        g('commit', '-q', '-m', 'change', '--allow-empty')
        return repo, base

    def task(self, issue):
        with open(self.state, encoding='utf-8') as f:
            return json.load(f)['tasks'][str(issue)]

    def events(self):
        with open(self.state, encoding='utf-8') as f:
            return json.load(f)['events']

    def cli(self, *argv):
        return run('--state', self.state, '--repo', self.repo, *argv)

    def accept(self, issue, reviewer='Codex'):
        rc, _, err = self.cli('start', '--issue', issue)
        assert rc == 0, err
        rc, _, err = self.cli('finish', '--issue', issue, '--to',
                              'implemented')
        assert rc == 0, err
        rc, _, err = self.cli('handoff', '--issue', issue,
                              '--outbox', self.outbox,
                              '--base', self.base_sha)
        assert rc == 0, err
        rc, _, err = self.cli('review', '--issue', issue,
                              '--verdict', 'accepted',
                              '--reviewer', reviewer,
                              '--evidence', f'issue-{issue}.patch')
        assert rc == 0, err

    def cleanup(self):
        self.dir.cleanup()


PLAN_DEP = [
    {'issue': 11, 'title': 'dependency task', 'dependencies': []},
    {'issue': 12, 'title': 'child task', 'dependencies': [11]},
]
PLAN_FREE = [
    {'issue': 21, 'title': 'independent a', 'dependencies': []},
    {'issue': 22, 'title': 'independent b', 'dependencies': []},
]
PLAN_RETRY = [
    {'issue': 31, 'title': 'retry task', 'dependencies': [],
     'max_attempts': 2, 'timeout_minutes': 0},
]


class DependencyGateTest(unittest.TestCase):

    def test_start_blocked_until_dependency_accepted(self):
        env = Env(PLAN_DEP, repo_files={'change.txt': 'change\n'})
        try:
            rc, _, err = env.cli('start', '--issue', 12)
            self.assertEqual(rc, 2)
            self.assertIn('dependency gate', err)
            self.assertIn('[11]', err)
            # the child task must be untouched
            self.assertEqual(env.task(12)['status'], 'pending')

            env.accept(11)
            rc, _, err = env.cli('start', '--issue', 12)
            self.assertEqual(rc, 0, err)
            self.assertEqual(env.task(12)['status'], 'running')
        finally:
            env.cleanup()

    def test_dependency_must_be_accepted_not_just_implemented(self):
        env = Env(PLAN_DEP, repo_files={'change.txt': 'change\n'})
        try:
            rc, _, _ = env.cli('start', '--issue', 11)
            self.assertEqual(rc, 0)
            rc, _, _ = env.cli('finish', '--issue', 11, '--to',
                               'implemented')
            self.assertEqual(rc, 0)
            rc, _, err = env.cli('start', '--issue', 12)
            self.assertEqual(rc, 2)
            self.assertIn('dependency gate', err)
        finally:
            env.cleanup()


class ReviewGateTest(unittest.TestCase):

    def setUp(self):
        self.env = Env(PLAN_FREE, repo_files={'change.txt': 'change\n'})
        rc, _, err = self.env.cli('start', '--issue', 21)
        assert rc == 0, err
        rc, _, err = self.env.cli('handoff', '--issue', 21,
                                  '--outbox', self.env.outbox,
                                  '--base', self.env.base_sha)
        assert rc == 2, 'handoff from running must be refused'

    def tearDown(self):
        self.env.cleanup()

    def _to_review_requested(self):
        rc, _, err = self.env.cli('finish', '--issue', 21, '--to',
                                  'implemented')
        self.assertEqual(rc, 0, err)
        rc, _, err = self.env.cli('handoff', '--issue', 21,
                                  '--outbox', self.env.outbox,
                                  '--base', self.env.base_sha)
        self.assertEqual(rc, 0, err)

    def test_acceptance_requires_reviewer_and_evidence(self):
        self._to_review_requested()
        rc, _, err = self.env.cli('review', '--issue', 21,
                                  '--verdict', 'accepted')
        self.assertEqual(rc, 2)
        self.assertIn('--reviewer', err)
        rc, _, err = self.env.cli('review', '--issue', 21,
                                  '--verdict', 'accepted',
                                  '--reviewer', 'Codex')
        self.assertEqual(rc, 2)
        self.assertIn('--evidence', err)
        self.assertEqual(self.env.task(21)['status'], 'review_requested')

    def test_self_review_refused(self):
        self._to_review_requested()
        rc, _, err = self.env.cli('review', '--issue', 21,
                                  '--verdict', 'accepted',
                                  '--reviewer', 'ZCode',
                                  '--evidence', 'patch')
        self.assertEqual(rc, 2)
        self.assertIn('self-approval', err)

    def test_accepted_with_explicit_record(self):
        self._to_review_requested()
        rc, _, err = self.env.cli('review', '--issue', 21,
                                  '--verdict', 'accepted',
                                  '--reviewer', 'Codex',
                                  '--evidence', 'issue-21.patch + evidence')
        self.assertEqual(rc, 0, err)
        task = self.env.task(21)
        self.assertEqual(task['status'], 'accepted')
        self.assertEqual(task['review']['reviewer'], 'Codex')

    def test_finish_cannot_reach_accepted(self):
        rc, _, err = self.env.cli('finish', '--issue', 21, '--to', 'accepted')
        self.assertEqual(rc, 2)          # argparse rejects the choice
        # status unchanged
        self.assertEqual(self.env.task(21)['status'], 'running')

    def test_needs_changes_then_rework(self):
        self._to_review_requested()
        rc, _, err = self.env.cli('review', '--issue', 21,
                                  '--verdict', 'needs_changes',
                                  '--reviewer', 'Codex',
                                  '--notes', 'tighten the acceptance')
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.env.task(21)['status'], 'needs_changes')
        rc, _, err = self.env.cli('start', '--issue', 21)
        self.assertEqual(rc, 2)          # needs_changes -> rework -> pending
        rc, _, err = self.env.cli('rework', '--issue', 21)
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.env.task(21)['status'], 'pending')
        self.assertEqual(self.env.task(21)['attempt'], 0)

    def test_reopen_withdraws_handoff_before_review(self):
        self._to_review_requested()
        rc, _, err = self.env.cli('reopen', '--issue', 21,
                                  '--reason', 'requirements changed')
        self.assertEqual(rc, 0, err)
        task = self.env.task(21)
        self.assertEqual(task['status'], 'pending')
        # attempt is kept: the work cycle continues, it does not restart
        self.assertEqual(task['attempt'], 1)
        self.assertIn('requirements changed', task['notes'][-1])
        self.assertTrue(any(e['event'] == 'reopen' for e in self.env.events()))
        # after reopen the task can run and hand off again
        rc, _, err = self.env.cli('start', '--issue', 21)
        self.assertEqual(rc, 0, err)
        rc, _, err = self.env.cli('finish', '--issue', 21, '--to',
                                  'implemented')
        self.assertEqual(rc, 0, err)
        rc, _, err = self.env.cli('handoff', '--issue', 21,
                                  '--outbox', self.env.outbox,
                                  '--base', self.env.base_sha)
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.env.task(21)['status'], 'review_requested')

    def test_reopen_requires_reason_and_review_requested(self):
        self._to_review_requested()
        rc, _, err = self.env.cli('reopen', '--issue', 21)
        self.assertEqual(rc, 2)          # argparse: --reason is required
        rc, _, err = self.env.cli('reopen', '--issue', 21, '--reason', 'x')
        self.assertEqual(rc, 0, err)
        # not review_requested anymore
        rc, _, err = self.env.cli('reopen', '--issue', 21, '--reason', 'x')
        self.assertEqual(rc, 2)
        self.assertIn('review_requested', err)

    def test_accepted_cannot_be_reopened(self):
        self._to_review_requested()
        rc, _, err = self.env.cli('review', '--issue', 21,
                                  '--verdict', 'accepted',
                                  '--reviewer', 'Codex',
                                  '--evidence', 'issue-21.patch')
        self.assertEqual(rc, 0, err)
        rc, _, err = self.env.cli('reopen', '--issue', 21,
                                  '--reason', 'change of mind')
        self.assertEqual(rc, 2)
        self.assertIn('accepted', err)
        self.assertEqual(self.env.task(21)['status'], 'accepted')


class RetryAndTimeoutTest(unittest.TestCase):

    def test_fail_retries_then_blocks(self):
        env = Env(PLAN_RETRY, repo_files={'change.txt': 'change\n'})
        try:
            for expected in ('pending', 'blocked'):
                rc, _, err = env.cli('start', '--issue', 31)
                self.assertEqual(rc, 0, err)
                rc, _, err = env.cli('fail', '--issue', 31)
                self.assertEqual(rc, 0, err)
                self.assertEqual(env.task(31)['status'], expected)
            # blocked by exhaustion: no more starts without unblock
            rc, _, err = env.cli('start', '--issue', 31)
            self.assertEqual(rc, 2)
            self.assertIn('unblock', err)
            rc, _, err = env.cli('unblock', '--issue', 31)
            self.assertEqual(rc, 0, err)
            self.assertEqual(env.task(31)['attempt'], 0)
        finally:
            env.cleanup()

    def test_expire_turns_missed_deadline_into_failed_attempt(self):
        env = Env(PLAN_RETRY, repo_files={'change.txt': 'change\n'})
        try:
            rc, _, err = env.cli('start', '--issue', 31)   # timeout 0 min
            self.assertEqual(rc, 0, err)
            self.assertTrue(os.path.exists(
                os.path.join(os.path.dirname(env.state), 'gpu.lock')))
            rc, _, err = env.cli('expire')
            self.assertEqual(rc, 0, err)
            task = env.task(31)
            self.assertEqual(task['status'], 'pending')
            self.assertEqual(task['attempt'], 1)
            self.assertTrue(any(e['event'] == 'timeout'
                                for e in env.events()))
            self.assertFalse(os.path.exists(
                os.path.join(os.path.dirname(env.state), 'gpu.lock')))
        finally:
            env.cleanup()


class GpuLockTest(unittest.TestCase):

    def test_mutual_exclusion_and_release(self):
        env = Env(PLAN_FREE, repo_files={'change.txt': 'change\n'})
        try:
            rc, _, err = env.cli('start', '--issue', 21)
            self.assertEqual(rc, 0, err)
            rc, _, err = env.cli('start', '--issue', 22)
            self.assertEqual(rc, 2)
            self.assertIn('GPU lock held', err)
            self.assertEqual(env.task(22)['status'], 'pending')
            rc, _, err = env.cli('finish', '--issue', 21, '--to',
                                 'implemented')
            self.assertEqual(rc, 0, err)
            rc, _, err = env.cli('start', '--issue', 22)
            self.assertEqual(rc, 0, err)
        finally:
            env.cleanup()

    def test_stale_lock_taken_over(self):
        env = Env(PLAN_FREE, repo_files={'change.txt': 'change\n'})
        try:
            # a pid that has already exited
            dead = subprocess.Popen(['true'])
            dead.wait()
            lock_path = os.path.join(os.path.dirname(env.state), 'gpu.lock')
            with open(lock_path, 'w', encoding='utf-8') as f:
                json.dump({'pid': dead.pid, 'issue': 99,
                           'heartbeat_at': cga.iso(cga.utcnow())}, f)
            rc, _, err = env.cli('start', '--issue', 21)
            self.assertEqual(rc, 0, err)
            self.assertTrue(any(e['event'] == 'lock-takeover'
                                for e in env.events()))
        finally:
            env.cleanup()

    def test_fresh_heartbeat_makes_live_lock_sticky(self):
        env = Env(PLAN_FREE, repo_files={'change.txt': 'change\n'})
        try:
            lock_path = os.path.join(os.path.dirname(env.state), 'gpu.lock')
            # live pid (this process) with a stale heartbeat -> takeover ok
            with open(lock_path, 'w', encoding='utf-8') as f:
                json.dump({'pid': os.getpid(), 'issue': 99,
                           'heartbeat_at': '2000-01-01T00:00:00+00:00'}, f)
            rc, _, err = env.cli('start', '--issue', 21)
            self.assertEqual(rc, 0, err)
        finally:
            env.cleanup()


class HandoffTest(unittest.TestCase):

    def test_handoff_artifacts_and_state(self):
        env = Env(PLAN_FREE, repo_files={'change.txt': 'change\n'})
        try:
            rc, _, _ = env.cli('start', '--issue', 21)
            self.assertEqual(rc, 0)
            rc, _, _ = env.cli('record-test', '--issue', 21,
                               '--command', 'python -m unittest tools',
                               '--observed', '12 tests OK')
            self.assertEqual(rc, 0)
            ev = os.path.join(env.base, 'ev.json')
            with open(ev, 'w', encoding='utf-8') as f:
                f.write('{}')
            rc, _, _ = env.cli('evidence', '--issue', 21, '--path', ev,
                               '--kind', 'benchmark-json')
            self.assertEqual(rc, 0)
            rc, _, _ = env.cli('finish', '--issue', 21, '--to', 'implemented')
            self.assertEqual(rc, 0)
            rc, _, err = env.cli('handoff', '--issue', 21,
                                 '--outbox', env.outbox,
                                 '--base', env.base_sha)
            self.assertEqual(rc, 0, err)

            request = json.load(open(os.path.join(env.outbox,
                                                  'review_request.json'),
                                     encoding='utf-8'))
            self.assertEqual(request['status'], 'review_requested')
            self.assertEqual(request['base_sha'], env.base_sha)
            paths = [f['path'] for f in request['changed_files']]
            self.assertIn('change.txt', paths)
            self.assertEqual(request['tests'][0]['command'],
                             'python -m unittest tools')
            self.assertEqual(len(request['evidence']), 1)
            self.assertTrue(request['evidence'][0]['sha256'])

            patch = open(os.path.join(env.outbox, 'issue-21.patch'),
                         encoding='utf-8').read()
            self.assertIn('+change', patch)
            body = open(os.path.join(env.outbox, 'issue-21-pr-body.md'),
                        encoding='utf-8').read()
            self.assertIn('Manual review required', body)
            self.assertEqual(env.task(21)['status'], 'review_requested')

            # handoff is one-shot from implemented
            rc, _, err = env.cli('handoff', '--issue', 21,
                                 '--outbox', env.outbox,
                                 '--base', env.base_sha)
            self.assertEqual(rc, 2)
        finally:
            env.cleanup()

    def test_handoff_requires_changes(self):
        env = Env(PLAN_FREE, repo_files={'change.txt': 'change\n'})
        try:
            rc, _, err = env.cli('handoff', '--issue', 21,
                                 '--outbox', env.outbox,
                                 '--base', env.base_sha)
            self.assertEqual(rc, 2)
            self.assertIn('requires implemented', err)
        finally:
            env.cleanup()

    def test_evidence_file_must_exist(self):
        env = Env(PLAN_FREE, repo_files={'change.txt': 'change\n'})
        try:
            rc, _, err = env.cli('evidence', '--issue', 21, '--path',
                                 os.path.join(env.base, 'nope.json'),
                                 '--kind', 'benchmark-json')
            self.assertEqual(rc, 2)
            self.assertIn('not found', err)
        finally:
            env.cleanup()


class ManifestTest(unittest.TestCase):

    def setUp(self):
        self.env = Env(PLAN_FREE, repo_files={'change.txt': 'change\n'})
        self.body = os.path.join(self.env.base, 'body.md')
        with open(self.body, 'w', encoding='utf-8') as f:
            f.write('body text\n')

    def tearDown(self):
        self.env.cleanup()

    def manifest(self, *extra, **kw):
        return self.env.cli('manifest', '--issue', 2,
                            '--outbox', self.env.outbox,
                            '--base-sha', kw.get('base', self.env.base_sha),
                            '--title', kw.get('title', 'issue 2 title'),
                            '--body-file', kw.get('body', self.body),
                            *extra)

    def test_manifest_exact_keys_and_content(self):
        rc, _, err = self.manifest()
        self.assertEqual(rc, 0, err)
        path = os.path.join(self.env.outbox, 'issue-2.json')
        with open(path, encoding='utf-8') as f:
            m = json.load(f)
        self.assertEqual(sorted(m), ['base_sha', 'body', 'files', 'issue',
                                     'title'])
        self.assertEqual(m['issue'], 2)
        self.assertEqual(m['base_sha'], self.env.base_sha)
        self.assertEqual([f['path'] for f in m['files']], ['change.txt'])
        self.assertEqual(m['files'][0]['content'], 'change\n')
        self.assertEqual(oct(os.stat(path).st_mode & 0o777), '0o644')

    def test_manifest_rejects_hidden_paths(self):
        env = Env(PLAN_FREE, repo_files={'change.txt': 'x'},
                  hidden_file=True)
        try:
            rc, _, err = env.cli('manifest', '--issue', 2,
                                 '--outbox', env.outbox,
                                 '--base-sha', env.base_sha,
                                 '--title', 't', '--body-file', self.body)
            self.assertEqual(rc, 2)
            self.assertIn('hidden path', err)
        finally:
            env.cleanup()

    def test_manifest_rejects_dirty_tree(self):
        with open(os.path.join(self.env.repo, 'uncommitted.txt'),
                  'w', encoding='utf-8') as f:
            f.write('dirty\n')
        rc, _, err = self.manifest()
        self.assertEqual(rc, 2)
        self.assertIn('dirty', err)
        rc, _, err = self.manifest('--allow-dirty')
        self.assertEqual(rc, 0, err)

    def test_manifest_rejects_oversized_fields(self):
        rc, _, err = self.manifest(title='x' * 201)
        self.assertEqual(rc, 2)
        self.assertIn('title', err)
        big = os.path.join(self.env.base, 'big.md')
        with open(big, 'w', encoding='utf-8') as f:
            f.write('x' * 20_001)
        rc, _, err = self.manifest(body=big)
        self.assertEqual(rc, 2)
        self.assertIn('body', err)

    def test_manifest_requires_full_sha(self):
        rc, _, err = self.manifest(base='deadbeef')
        self.assertEqual(rc, 2)
        self.assertIn('40-hex', err)

    def test_manifest_expanded_file_count_boundary(self):
        for count, expected in ((67, 0), (100, 0), (101, 2)):
            env = Env(PLAN_FREE, repo_files={f'f{i}.txt': 'x\n' for i in range(count)})
            try:
                rc, _, err = env.cli('manifest', '--issue', 2,
                                     '--outbox', env.outbox,
                                     '--base-sha', env.base_sha,
                                     '--title', 'bounded publication', '--body-file', self.body)
                self.assertEqual(rc, expected, err)
                if rc:
                    self.assertIn('100-file', err)
            finally:
                env.cleanup()


class DailyReportTest(unittest.TestCase):

    def test_fields_and_honesty_flags(self):
        env = Env(PLAN_FREE, repo_files={'change.txt': 'change\n'})
        try:
            rc, _, _ = env.cli('start', '--issue', 21)
            self.assertEqual(rc, 0)
            rc, _, _ = env.cli('finish', '--issue', 21, '--to',
                               'implemented')
            self.assertEqual(rc, 0)
            rc, out, err = env.cli('daily-report')
            self.assertEqual(rc, 0, err)
            report = json.loads(out)
            for key in ('schema', 'generated_at_utc', 'issues',
                        'recent_events', 'gpu_lock', 'in_review', 'accepted',
                        'blocked', 'next_actions', 'unverified'):
                self.assertIn(key, report)
            # an implemented task with no evidence must be flagged, not hidden
            self.assertIn({'issue': 21, 'note': 'no evidence recorded yet'},
                          report['unverified'])
            self.assertEqual(report['gpu_lock'], None)   # released on finish
        finally:
            env.cleanup()

    def test_next_action_respects_dependency_gate(self):
        env = Env(PLAN_DEP, repo_files={'change.txt': 'change\n'})
        try:
            rc, out, _ = env.cli('daily-report')
            report = json.loads(out)
            starts = [a['issue'] for a in report['next_actions']
                      if a['action'].startswith('start')]
            self.assertEqual(starts, [11])   # 12 is gated on 11
        finally:
            env.cleanup()


if __name__ == '__main__':
    unittest.main()
