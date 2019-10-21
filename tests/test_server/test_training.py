import torch
import pytest
import time
import queue
import threading
from torch import multiprocessing as mp
from unittest import mock

from tests.data.tiny_models import TinyConvNet2d
from tiktorch.rpc.mp import MPClient, Shutdown, create_client
from tiktorch.server.training import ITraining
from tiktorch.server.training.base import TrainingProcess, ConfigBuilder, run
from tiktorch.server import training
from tiktorch.server.training.worker.base import Engine
from tiktorch.server.training.worker import commands, State
from tiktorch.tiktypes import TikTensor, TikTensorBatch
from tiktorch import configkeys as confkeys


def test_initialization(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    tp = TrainingProcess(config=config, model=model)
    tp.shutdown()


def test_set_devices(tiny_model_2d):
    config = tiny_model_2d["config"]
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)

    tp = TrainingProcess(config=config, model=model)
    tp.set_devices([torch.device("cpu")])
    tp.shutdown()


def test_training(tiny_model_2d):
    config = tiny_model_2d["config"]
    config["num_iterations_per_update"] = 10
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    training = TrainingProcess(config=config, model=model)
    try:
        training.set_devices([torch.device("cpu")])
        data = TikTensorBatch(
            [
                TikTensor(torch.zeros(in_channels, 15, 15), ((1,), (1,))),
                TikTensor(torch.ones(in_channels, 15, 15), ((2,), (2,))),
            ]
        )
        labels = TikTensorBatch(
            [
                TikTensor(torch.ones(in_channels, 15, 15, dtype=torch.uint8), ((1,), (1,))),
                TikTensor(torch.full((in_channels, 15, 15), 2, dtype=torch.uint8), ((2,), (2,))),
            ]
        )
        training.update_dataset("training", data, labels)
        training.resume_training()
        st = training.get_state()
        training.wait_for_idle().result()
        st = training.get_state()
    finally:
        training.shutdown()


from numpy.testing import assert_equal


def _assert_equal(v1, v2):
    if isinstance(v1, dict) and isinstance(v2, dict):
        assert v1.keys() == v2.keys()
        for key in v1:
            _assert_equal(v1[key], v2[key])

    elif isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        assert_equal(v1.numpy(), v2.numpy())

    else:
        assert v1 == v2


def test_training_restore_optimizer_state(tiny_model_2d):
    config = tiny_model_2d["config"]
    config["num_iterations_per_update"] = 10
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    training = TrainingProcess(config=config, model=model)
    try:
        training.set_devices([torch.device("cpu")])
        data = TikTensorBatch(
            [
                TikTensor(torch.zeros(in_channels, 15, 15), ((1,), (1,))),
                TikTensor(torch.ones(in_channels, 15, 15), ((2,), (2,))),
            ]
        )
        labels = TikTensorBatch(
            [
                TikTensor(torch.ones(in_channels, 15, 15, dtype=torch.uint8), ((1,), (1,))),
                TikTensor(torch.full((in_channels, 15, 15), 2, dtype=torch.uint8), ((2,), (2,))),
            ]
        )
        training.update_dataset("training", data, labels)
        before_training_state = training.get_state()
        training.resume_training()
        training.wait_for_idle().result()

        after_training_state = training.get_state()
        assert before_training_state.optimizer_state != after_training_state.optimizer_state

        new_training = TrainingProcess(config=config, model=model, optimizer_state=after_training_state.optimizer_state)
        assert len(training.trainer.optimizer.state_dict()["state"]) == len(
            new_training.trainer.optimizer.state_dict()["state"]
        )

        zipped = zip(
            training.trainer.optimizer.state_dict()["state"].values(),
            new_training.trainer.optimizer.state_dict()["state"].values(),
        )
        for t1, t2 in zipped:
            _assert_equal(t1, t2)
            # print(t1)
            # assert_equal(t1.numpy(), t2.numpy())

        # new_state = new_training.get_state()

        # assert after_training_state.optimizer_state == new_state.optimizer_state
    finally:
        training.shutdown()


def test_training_in_proc(tiny_model_2d, log_queue):
    config = tiny_model_2d["config"]
    config["num_iterations_per_update"] = 10
    in_channels = config["input_channels"]
    model = TinyConvNet2d(in_channels=in_channels)
    handler_conn, training_conn = mp.Pipe()
    p = mp.Process(target=run, kwargs={"conn": training_conn, "model": model, "config": config, "log_queue": log_queue})
    p.start()
    client = create_client(ITraining, handler_conn)
    try:
        client.set_devices([torch.device("cpu")])
        data = TikTensorBatch(
            [
                TikTensor(torch.zeros(in_channels, 15, 15), ((1,), (1,))),
                TikTensor(torch.ones(in_channels, 9, 9), ((2,), (2,))),
            ]
        )
        labels = TikTensorBatch(
            [
                TikTensor(torch.ones(in_channels, 15, 15, dtype=torch.uint8), ((1,), (1,))),
                TikTensor(torch.full((in_channels, 9, 9), 2, dtype=torch.uint8), ((2,), (2,))),
            ]
        )
        client.update_dataset("training", data, labels)
        client.resume_training()
    finally:
        client.shutdown()


class TestTrainingWorkerEngine:
    class DummyCmd(commands.ICommand):
        def execute(self, ctx):
            pass

    class DummyTrainer:
        def __init__(self):
            self.iteration_count = 0
            self.max_num_iterations = 0
            self._break_cb = None
            self._devs = []

        def set_break_callback(self, cb):
            self._break_cb = cb

        def move_to(self, devices):
            self._devs = devices

        def set_max_num_iterations(self, val):
            self.max_num_iterations = val

        def stop_fitting(self, max_num_iterations=None, max_num_epochs=None):
            return self._break_cb and self._break_cb() or self.iteration_count >= self.max_num_iterations

        def fit(self):
            while not self.stop_fitting():
                self.iteration_count += 1
                time.sleep(0.01)

    @pytest.fixture
    def trainer(self):
        return self.DummyTrainer()

    @pytest.fixture
    def worker(self, trainer):
        return Engine(trainer)

    @pytest.fixture
    def worker_thread(self, worker):
        t = threading.Thread(target=worker.run)
        t.start()
        yield t
        worker.send_command(commands.StopCmd())
        t.join()

    def test_not_running_worker_has_stopped_status(self, worker):
        assert State.Stopped == worker.state

    def test_started_worker_has_idle_status(self, worker, worker_thread):
        cmd = self.DummyCmd().awaitable
        worker.send_command(cmd)
        cmd.wait()

        assert State.Paused == worker.state

    def test_resuming_transitions_to_idle_with_no_devices(self, worker, worker_thread):
        cmd = commands.ResumeCmd().awaitable
        worker.send_command(cmd)
        cmd.wait()

        assert State.Idle == worker.state

    def test_transition_to_running(self, worker, worker_thread):
        cmd = commands.ResumeCmd()
        worker.send_command(cmd)

        add_work = commands.SetMaxNumberOfIterations(1000).awaitable
        worker.send_command(add_work)
        add_work.wait()

        assert State.Idle == worker.state

        add_device = commands.SetDevicesCmd([torch.device("cpu")]).awaitable
        worker.send_command(add_device)

        add_device.wait()

        assert State.Running == worker.state

        remove_device = commands.SetDevicesCmd([])
        awaitable_remove = remove_device.awaitable
        worker.send_command(awaitable_remove)
        awaitable_remove.wait()
        assert [torch.device("cpu")] == remove_device.result

        assert State.Idle == worker.state

    def test_exception_during_train_should_transition_to_paused(self, worker, worker_thread, trainer):
        fit_called = threading.Event()

        def _exc():
            fit_called.set()
            raise Exception()

        trainer.fit = _exc

        cmd = commands.ResumeCmd()
        worker.send_command(cmd)

        add_work = commands.SetMaxNumberOfIterations(1000).awaitable
        worker.send_command(add_work)
        add_work.wait()

        assert State.Idle == worker.state

        assert not fit_called.is_set()
        add_device = commands.SetDevicesCmd([torch.device("cpu")]).awaitable
        worker.send_command(add_device)
        add_device.wait()

        fit_called.wait()
        time.sleep(0.2)  # FIXME: Find a better way to wait for pause event with timeout
        assert State.Paused == worker.state


class TestConfigBuilder:
    @pytest.mark.parametrize(
        "key,expected_default,provided_value",
        [
            (confkeys.NUM_ITERATIONS_MAX, 0, 100),
            (confkeys.NUM_ITERATIONS_PER_UPDATE, 1, 4),
            (confkeys.OPTIMIZER_CONFIG, {"method": "Adam"}, {"method": "Adagrad"}),
        ],
    )
    def test_config_defauts(self, key, expected_default, provided_value):
        config = ConfigBuilder.build({"training": {}})

        assert expected_default == config[key]

        config = ConfigBuilder.build({"training": {key: provided_value}})
        assert provided_value == config[key]

    def test_config_with_default_loss(self):
        config = ConfigBuilder.build({"training": {}})

        loss_conf = config.get("criterion_config")
        assert loss_conf
        assert isinstance(loss_conf["method"].criterion, torch.nn.MSELoss)

    def test_config_with_specified_loss(self):
        config = ConfigBuilder.build({"training": {confkeys.LOSS_CRITERION_CONFIG: {"method": "CrossEntropyLoss"}}})
        loss_conf = config.get("criterion_config")
        assert loss_conf
        assert isinstance(loss_conf["method"].criterion, torch.nn.CrossEntropyLoss)


from hamcrest.core.base_matcher import BaseMatcher
from hamcrest.core.helpers.hasmethod import hasmethod


class IsGivenDayOfWeek(BaseMatcher):
    def __init__(self, day):
        self.day = day  # Monday is 0, Sunday is 6

    def _matches(self, item):
        print("MACHES CALLED")
        if not hasmethod(item, "weekday"):
            return False
        return item.weekday() == self.day

    def describe_to(self, description):
        day_as_string = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        description.append_text("calendar date falling on ").append_text(day_as_string[self.day])


import numpy


class Comparator:
    _registry = {}

    @classmethod
    def register(cls, left_type, right_type, comparator):
        cls._registry[(left_type, right_type)] = comparator

    def __init__(self, left, right):
        self._left = left
        self._right = right

    def evaluate(self, ctx):
        compare_fn = self._registry.get((type(self._left), type(self._right)))

        if compare_fn is not None:
            ctx.evaluate(lambda: compare_fn(self._left, self._right))
        else:
            assert self._left == self._right


def _numpy_compare(val, other):
    return numpy.array_equal(val, other)


Comparator.register(numpy.ndarray, numpy.ndarray, _numpy_compare)


class Context:
    def __init__(self):
        self.negated = False

    def evaluate(self, bool_fn):
        if not self.negated:
            assert bool_fn()
        else:
            assert not bool_fn()


class expect:
    def __init__(self, val):
        self._val = val
        self._ctx = Context()

    @property
    def to(self):
        return self

    @property
    def be(self):
        return self

    @property
    def have(self):
        return self

    @property
    def not_(self):
        self._ctx.negated = True
        return self

    def equal(self, other):
        Comparator(self._val, other).evaluate(self._ctx)
        return self

    def a(self, type_):
        assert isinstance(self._val, type_)
        return self

    an = a


def test_1():
    expect(numpy.array([1, 2])).to.not_.be.equal(numpy.array([1, 2]))
    expect(1).to.be.equal(1)
    expect(1).to.be.an(int)
