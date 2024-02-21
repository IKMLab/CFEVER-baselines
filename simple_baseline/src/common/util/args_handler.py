# -*- coding: utf-8 -*-
"""
Arguments handler module
===================

This module provides functions for parsing arguments.


.. codeauthor:: Yi-Ting Li <yt.li.public@gmail.com>
.. codeauthor:: Ying-Jia Lin <yingjia.lin.public@gmail.com>
"""
import argparse
import json
from dataclasses import dataclass, field

from typing import Any, Iterable, Optional, Tuple, Union


@dataclass
class Argument(dict):
    """Argument wraps argparse.ArgumentParser.add_argument arguments"""

    name_or_flags: str
    type: Optional[type] = None
    const: Any = None
    default: Any = None
    required: bool = False
    action: Optional[str] = None
    nargs: Optional[int] = None
    metavar: Optional[str] = None
    choices: Optional[Iterable] = None
    help: Optional[str] = None
    metavar: Optional[Union[str, Tuple[str, ...]]] = None
    dest: Optional[str] = None
    kwargs: Optional[dict] = field(default_factory=dict)


@dataclass
class Arguments:
    """Arguments handler for argparse

    Example:
        >>> class BERTArguments(Arguments):
        >>>     mode = Argument(
        >>>         "--mode",
        >>>         required=True,
        >>>         choices=["train", "eval"],
        >>>         help="train or eval",
        >>>     )
        >>>     lr = Argument(
        >>>         "--lr",
        >>>         default=2e-5,
        >>>         type=float,
        >>>         help="learning rate",
        >>>     )
        >>> bert_args = BERTArguments.parse_args()
    """

    def __repr__(self) -> str:
        return str(self.__dict__.__annotations__)

    @classmethod
    def parse_args(cls) -> argparse.ArgumentParser:
        """parse_args parses the arguments and returns the `argparse.ArgumentParser`

        Returns:
            argparse.ArgumentParser: unparsed arguments
        """
        arg = argparse.ArgumentParser()

        # iterate through the class attributes
        for k, v in cls.__dict__.items():

            # only add the arguments that are not private
            if not k.startswith("__"):
                # get the name_or_flags to check if the argument is required
                name_or_flags = v.__dict__.pop("name_or_flags")

                # if the argument is required, remove the required flag
                if "--" in name_or_flags:
                    v.__dict__.pop("required", None)

                # second layer of kwargs
                kwargs = v.__dict__.pop("kwargs", {})

                # add the argument
                arg.add_argument(
                    name_or_flags,
                    **v.__dict__,
                    **kwargs,
                )

        return arg.parse_args()


def save_args_to_json(args: argparse.Namespace, json_path: str):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    Return:
        a saved json file containing hyperparameters
    """
    dict_of_params = vars(args)
    with open(json_path, "w") as f:
        dict_of_params = {k: v for k, v in dict_of_params.items()}
        json.dump(dict_of_params, f, indent=4)
