from __future__ import annotations

import logging
import enum

from inferno.trainers import Trainer as InfernoTrainer

from torch.utils.data import DataLoader

from tiktorch.server.datasets import DynamicDataLoaderWrapper, DynamicDataset

logger = logging.getLogger(__name__)


@enum.unique
class DatasetType(enum.Enum):
    Training = "train"
    Validation = "validation"
    Test = "test"


class ITrainer:
    def set_break_callback(self, callback) -> None:
        """
        This callback called after each training iteration to check if training should halt
        """
        raise NotImplementedError

    def get_dataset(self, name: DatasetType) -> DynamicDataset:
        """
        Return dataset for modification
        """
        raise NotImplementedError

    @property
    def max_num_iterations(self) -> int:
        raise NotImplementedError

    def set_max_num_iterations(self, num: int) -> None:
        raise NotImplementedError

    def move_to(self, devices):
        """
        Move trainer to specified devices
        """
        raise NotImplementedError


class TrainerBuilder:
    def build(self, recipe: IRecipe) -> ITrainer:
        """
        Returns fully configured trainer with model and optimizer loaded
        """
        return recipe.make()


class IRecipe:
    def make(self) -> ITrainer:
        raise NotImplementedError


class InfernoConfRecipe(IRecipe):
    def __init__(self, config, datasets):
        self._config = config
        self._datasets = datasets

    def make(self):
        return TikTrainer.build(self._config)


class TikTrainer(InfernoTrainer):
    _ALIASES = {"training": "train", "validation": "validate"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._break_cb = None
        self._dataset_by_name = {}

    def set_break_callback(self, callback):
        self._break_cb = callback

    def get_dataset(self, name: str) -> DynamicDataset:
        return self._dataset_by_name[name]

    @property
    def max_num_iterations(self):
        return self._max_num_iterations

    def stop_fitting(self, max_num_iterations=None, max_num_epochs=None):
        if self._break_cb and self._break_cb():
            return True
        else:
            return super().stop_fitting(max_num_iterations=max_num_iterations, max_num_epochs=max_num_epochs)

    @classmethod
    def build(cls, *args, dataset_by_name, **kwargs):
        trainer = super().build(*args, **kwargs)

        trainer._dataset_by_name = dataset_by_name

        for name, dataset in dataset_by_name.items():
            name = cls._ALIASES.get(name, name)
            loader = DataLoader(dataset=dataset)
            trainer.bind_loader(name, DynamicDataLoaderWrapper(loader))

        return trainer

    def move_to(self, devices):
        if devices.base_device == "cpu":
            self.cpu()
        elif devices.base_device == "cuda":
            self.cuda(devices=[d.index for d in devices])
        else:
            raise ValueError(f"Unknown device type {devices.base_device}")

        # make sure optimizer states are on correct device
        for k in self.optimizer.state.keys():
            param_state = self.optimizer.state[k]
            for p in param_state.keys():
                try:
                    if not isinstance(param_state[p], int):
                        param_state[p] = param_state[p].to(devices.base_device)
                except Exception as e:
                    self.logger.exception("Failed to move optimizer to %s", devices)

    def create_optimizer(config: Dict, model, device, optimizer_state: bytes) -> Optional[torch.optim.Optimizer]:
        try:
            kwargs = config.copy()
            optimizer_class: Type[torch.optim.Optimizer] = getattr(torch.optim, kwargs.pop("method"))
            optimizer = optimizer_class(model.parameters(), **kwargs)
            try:
                optimizer.load_state_dict(torch.load(io.BytesIO(optimizer_state), map_location=device))
            except Exception as e:
                self.logger.warning(
                    "Could not load optimizer state due to %s.\nCreating new optimizer from %s", e, config
                )
            else:
                self.logger.info("restored optimizer state")
        except Exception as e:
            self.logger.exception(e)
            return None
        else:
            return optimizer
