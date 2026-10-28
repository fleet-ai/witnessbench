"""Canonical parity across a real JSON/HTTP boundary, without model/network services."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from golden import STORE, ScriptedPolicy
from witness_agent import EpisodeAborted, EpisodeLimits, run_episode, write_trace
from witness_agent.episode import Completion
from witness_env import GAME_IDS, TRAIN_R1_IDS, TRAIN_R2_IDS, make_game
from witness_env.client import LocalGameClient
from witness_game import RemoteGameClient
from witness_game.service import GameService
from witness_game.wire import decode_frame, encode_frame


@pytest.fixture
def remote():
    services = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            service = services[request['game']]
            result = service.action(**request['arguments'])
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def connect(game_id):
        service = services[game_id] = GameService(LocalGameClient(make_game(game_id)))
        def call(name, arguments):
            assert name == 'game_action'
            req = Request(f'http://127.0.0.1:{server.server_port}',
                          json.dumps({'game': game_id, 'arguments': arguments}).encode(),
                          {'Content-Type': 'application/json'})
            with urlopen(req, timeout=10) as response:
                return json.load(response)
        call.endpoint = f'http://127.0.0.1:{server.server_port}'
        return RemoteGameClient(call, service.describe())

    yield connect
    server.shutdown()
    server.server_close()
    worker.join()


@pytest.mark.parametrize('name', STORE.episode_names())
def test_stored_episodes_have_identical_prompts_results_and_trace_bytes(name, remote, tmp_path):
    episode = STORE.episode(name)
    limits = EpisodeLimits(episode.max_levels,
                           episode.max_reflection_calls if episode.wiring == 'eval' else None)
    results = []
    for game in (make_game(episode.game_id), remote(episode.game_id)):
        policy = ScriptedPolicy(episode)
        results.append(run_episode(game, seed=episode.seed, limits=limits, policy=policy))
        assert policy.mismatch is None
        assert policy.calls_made == len(episode.calls) + (episode.final_prompt is not None)
    assert results[0] == results[1]
    for i, result in enumerate(results):
        write_trace(result, tmp_path / str(i))
    assert {p.name: p.read_bytes() for p in (tmp_path / '0').iterdir()} == {
        p.name: p.read_bytes() for p in (tmp_path / '1').iterdir()}


@pytest.mark.parametrize('game_id', GAME_IDS + TRAIN_R1_IDS + TRAIN_R2_IDS)
def test_all_public_games_preserve_truncation_fallback_and_resets(game_id, remote):
    tape = [Completion('<plan>1</plan>', 'length', {}, 'test', 'tape', 1, 0.0),
            '<plan>0</plan>', '', '<plan>1,2,5,0,0,4</plan>', None]
    results = []
    for game in (make_game(game_id), remote(game_id)):
        answers = iter(tape)
        results.append(run_episode(game, seed=37, limits=EpisodeLimits(20, 8),
                                   policy=lambda s, u: next(answers)))
    assert results[0] == results[1]
    assert results[1].calls[0].meta['truncated'] is True


def test_duplicate_delivery_is_acknowledged_but_never_executed_twice():
    service = GameService(LocalGameClient(make_game('tw01')))
    request = dict(session_id=service.session_id, sequence=0, action=0)
    first = service.action(**request)
    assert service.action(**request) == first
    with pytest.raises(ValueError, match='different action'):
        service.action(**{**request, 'action': 1})
    service.action(**{**request, 'sequence': 1, 'action': 1})
    with pytest.raises(ValueError, match='out of order'):
        service.action(**request)
    restarted = GameService(LocalGameClient(make_game('tw01')))
    with pytest.raises(ValueError, match='invalid game request'):
        restarted.action(**request)


def test_failed_observer_after_action_prevents_replay():
    class BrokenObserver(LocalGameClient):
        calls = 0
        def step(self, action):
            self.calls += 1
            super().step(action)
            raise OSError('recording failed after action')
    game = BrokenObserver(make_game('tw01'))
    service = GameService(game)
    request = dict(session_id=service.session_id, sequence=0, action=0)
    with pytest.raises(OSError):
        service.action(**request)
    with pytest.raises(RuntimeError, match='uncertain'):
        service.action(**request)
    assert game.calls == 1


def test_lost_reply_aborts_partial_episode_without_another_action():
    service = GameService(LocalGameClient(make_game('tw01')))
    delivered = []
    def lost_reply(name, arguments):
        delivered.append(arguments)
        service.action(**arguments)
        raise TimeoutError('reply lost')
    client = RemoteGameClient(lost_reply, service.describe())
    with pytest.raises(EpisodeAborted) as error:
        run_episode(client, seed=0, limits=EpisodeLimits(1, 1), policy=lambda s, u: '')
    assert error.value.partial.stop_reason == 'aborted'
    with pytest.raises(RuntimeError, match='uncertain'):
        client.reset()
    assert len(delivered) == 1


def test_wire_rejects_boolean_counts_unknown_fields_and_bad_positions():
    frame = encode_frame(LocalGameClient(make_game('tw01')).reset())
    for value in ({**frame, 'levels_completed': True}, {**frame, 'extra': 'engine'},
                  {**frame, 'truth': {**frame['truth'], 'path': [[1]]}}):
        with pytest.raises(ValueError):
            decode_frame(value)


def test_remote_harness_imports_and_executes_without_game_or_engine(tmp_path):
    import shutil
    root = Path(__file__).resolve().parents[1]
    for package in ('witness_agent', 'witness_game'):
        shutil.copytree(root / package, tmp_path / package, ignore=shutil.ignore_patterns('__pycache__'))
    frame = encode_frame(LocalGameClient(make_game('tw01')).reset())
    (tmp_path / 'frame.json').write_text(json.dumps(frame))
    script = r'''
import json, sys
from witness_agent import run_episode, EpisodeLimits
from witness_game import RemoteGameClient
from pathlib import Path
from witness_agent.episode import Task, load_tasks
row = dict(prompt='tw01', metadata=dict(game_id='tw01', max_levels=1, max_reflection_calls=3))
task = Task.from_row(row, known_game_ids=('tw01',))
Path('tasks.jsonl').write_text(json.dumps(row) + '\n')
assert load_tasks(Path('tasks.jsonl'), known_game_ids=('tw01',)) == (task,)
try:
    Task.from_row(row, known_game_ids=('tw02',))
except ValueError:
    pass
else:
    raise AssertionError('unadmitted game accepted')
frame = json.load(open('frame.json'))
def call(name, args):
    return dict(session_id='isolated', sequence=args['sequence'], frame=frame)
client = RemoteGameClient(call, dict(protocol='witness.game.v1', session_id='isolated', game_id='tw01'))
result = run_episode(client, seed=0, limits=EpisodeLimits(1), policy=lambda s, u: None)
assert result.stop_reason == 'policy_stop'
assert not any(m.split('.')[0] in ('arcengine', 'witness_env', 'numpy') for m in sys.modules)
'''
    subprocess.run([sys.executable, '-S', '-c', script], cwd=tmp_path, check=True,
                   env={'PATH': str(Path(sys.executable).parent)})


def test_complete_episode_in_engine_free_process_over_http(remote, tmp_path):
    import shutil
    client = remote('tw01')
    # Recover only public transport data, never an engine object, into the subprocess.
    description = dict(protocol='witness.game.v1', session_id=client.session_id, game_id=client.game_id)
    endpoint = client.call.endpoint
    root = Path(__file__).resolve().parents[1]
    for package in ('witness_agent', 'witness_game'):
        shutil.copytree(root / package, tmp_path / package, ignore=shutil.ignore_patterns('__pycache__'))
    tape = ['<plan>0</plan>', '<plan>1,1,4,2,2,4,1,1,5</plan>', '<plan>1,0,0</plan>', '', None]
    (tmp_path / 'input.json').write_text(json.dumps(dict(endpoint=endpoint, description=description, tape=tape)))
    script = r'''
import json, sys
from urllib.request import Request, urlopen
from witness_agent import run_episode, EpisodeLimits, write_trace
from witness_game import RemoteGameClient
config = json.load(open('input.json'))
def call(name, arguments):
    request = Request(config['endpoint'], json.dumps(dict(game='tw01', arguments=arguments)).encode())
    with urlopen(request, timeout=10) as response:
        return json.load(response)
answers = iter(config['tape'])
result = run_episode(RemoteGameClient(call, config['description']), seed=37,
                     limits=EpisodeLimits(10, 8), policy=lambda s, u: next(answers))
from pathlib import Path
write_trace(result, Path('remote'))
assert not any(m.split('.')[0] in ('arcengine', 'witness_env', 'numpy') for m in sys.modules)
'''
    subprocess.run([sys.executable, '-S', '-c', script], cwd=tmp_path, check=True,
                   env={'PATH': str(Path(sys.executable).parent)})
    answers = iter(tape)
    result = run_episode(make_game('tw01'), seed=37, limits=EpisodeLimits(10, 8),
                         policy=lambda s, u: next(answers))
    write_trace(result, tmp_path / 'local')
    assert result.levels_completed == 1
    assert {p.name: p.read_bytes() for p in (tmp_path / 'local').iterdir()} == {
        p.name: p.read_bytes() for p in (tmp_path / 'remote').iterdir()}
