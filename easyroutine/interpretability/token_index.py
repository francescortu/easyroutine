# This file contains the TokenIndex class which is used to categorize tokens and get the index of tokens in the string tokens
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal
import random
import yaml
import importlib.resources
from easyroutine.logger import logger

TokenType = Union[str, int, Tuple[int, int]]  # New type for tokens
SpecialTokenType = Union[str, List[str]]  # New type for special tokens


# Load the YAML configuration file
def load_config() -> dict:
    with importlib.resources.open_text(
        "easyroutine.interpretability.config", "config.yaml"
    ) as file:
        return yaml.safe_load(file)


config = load_config()

SUPPORTED_MODELS = config["models"]
SUPPORTED_TOKENS = config["token_position"]


class TokenIndex:
    r"""
    TokenIndex is one of the core class of the interpretability module.
    It is used to find the right indexes that correspond to the tokens in the input of the model.
    In this way we are able to extract the right hidden states and attention weights, based on the tokens we are interested in.
    It support mixed modalities inputs, with both text and images.

    """

    def __init__(
        self,
        model_name: str,
        pivot_positions: Optional[List[int]] = None,
        pivot_tokens: Optional[List[str]] = None,
    ):
        r"""
        Args:
            model_name: str (required): the name of the model
            pivot_positions: List[int] (optional): a list of integers that represent the positions where to split the tokens.
            pivot_tokens: List[str] (optional): a list of strings that represent the tokens where to split the tokens.


        The pivot_positions and pivot_tokens are mutually exclusive.
        The idea of the split is the following. Immagine to have an input string of tokens like this: ["I", "love", "cats", "and", "dogs". "What", "about", "you?"]
        Then, i want to extract/ablate/intervene on the second sentence. I can do it by specifying the pivot_positions=[5] or pivot_tokens=["What"].
        In this way, the tokens will be split in two groups: ["I", "love", "cats", "and"] and ["dogs", "What", "about", "you?"] with names "inputs-partition-0" and "inputs-partition-1".
        """
        self.model_name = model_name
        self.pivot_tokens = pivot_tokens
        self.pivot_positions = sorted(pivot_positions) if pivot_positions else []

    def find_occurrences(self, lst: List[str], target: str) -> List[int]:
        return [i for i, x in enumerate(lst) if x == target]

    def categorize_tokens(
        self,
        string_tokens: List[str],
        include_delimiters_in_image: Optional[bool] = None,
    ) -> Dict[str, List[int]]:
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError("Unsupported model_name")

        # Unpack model tokens:
        # start_image_token: token marking the beginning of an image section
        # special: special tokens or sequences that might occur inside an image section
        # end_image_token: token marking the end of an image section
        start_image_token, special, end_image_token = SUPPORTED_MODELS[self.model_name]

        # Convert special to list if it's a single token
        if isinstance(special, str):
            special = [special]
        # Convert special tokens to list of lists for uniform processing
        if special is None:
            special_sequences = []
        else:
            special_sequences = [s if isinstance(s, list) else [s] for s in special]

        image_start_tokens = []
        image_end_tokens = []
        image_tokens = []
        last_line_image_tokens = []
        text_tokens = []
        special_tokens = []

        # Case 1: When the start and end markers differ.
        if start_image_token is None or start_image_token != end_image_token:
            in_image_sequence = False
            i = 0
            while i < len(string_tokens):
                token = string_tokens[i]
                if token == start_image_token and not in_image_sequence:
                    in_image_sequence = True
                    image_start_tokens.append(i)
                    i += 1
                    continue

                if in_image_sequence and token == end_image_token:
                    in_image_sequence = False
                    image_end_tokens.append(i)
                    last_line_image_tokens.append(i - 1)
                    i += 1
                    continue

                if in_image_sequence:
                    # Check for special token sequences
                    sequence_found = False
                    for seq in special_sequences:
                        if i + len(seq) <= len(string_tokens):
                            # Check if the current position matches the sequence
                            if string_tokens[i : i + len(seq)] == seq:
                                special_tokens.extend(range(i, i + len(seq)))
                                i += len(seq)
                                sequence_found = True
                                break
                    if sequence_found:
                        continue

                    image_tokens.append(i)
                else:
                    text_tokens.append(i)
                i += 1
        # Case 2: When the start and end markers are identical.
        else:
            # Collect all indices where the marker appears.
            marker_indices = [
                i for i, token in enumerate(string_tokens) if token == start_image_token
            ]
            if marker_indices:
                # Always record the first marker as image_start and the last as image_end.
                image_start_tokens.append(marker_indices[0])
                image_end_tokens.append(marker_indices[-1])
                # If no explicit flag was provided, inspect the content between the markers:
                if include_delimiters_in_image is None:
                    # If any token between the first and last marker is not the marker itself,
                    # we assume the markers are used as delimiters.
                    if any(
                        string_tokens[i] != start_image_token
                        for i in range(marker_indices[0] + 1, marker_indices[-1])
                    ):
                        include_delimiters_in_image = False
                    else:
                        include_delimiters_in_image = True

                if include_delimiters_in_image:
                    # If markers are part of the image content, include every occurrence.
                    image_tokens.extend(marker_indices)
                else:
                    # Otherwise, consider only the tokens strictly between the first and last markers.
                    if len(marker_indices) > 2:
                        image_tokens.extend(marker_indices[1:-1])
                    # (If there are exactly two markers, there is no “internal” image token.)
                # For this branch, define last_line_image as the position immediately before the last marker.
                last_line_image_tokens.append(marker_indices[-1] - 1)

            # Classify all tokens that are not the marker as text.
            for i, token in enumerate(string_tokens):
                if token != start_image_token:
                    text_tokens.append(i)
                # Optionally, if special tokens may occur outside the image region:
                if special and token in special:
                    special_tokens.append(i)

        tokens_group, positions_group = self.group_tokens(string_tokens)
        position_dict = {
            f"inputs-partition-{i}": positions_group[i] for i in positions_group
        }

        return {
            "image_start": image_start_tokens,
            "image_end": image_end_tokens,
            "image": image_tokens,
            "last_line_image": last_line_image_tokens,
            "text": text_tokens,
            "special": special_tokens,
            "all": list(range(len(string_tokens))),
            **position_dict,
        }

    def group_tokens(
        self, string_tokens: List[str]
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[int]]]:
        if self.pivot_tokens:
            return self.group_tokens_by_pivot_tokens(string_tokens)
        elif self.pivot_positions:
            return self.group_tokens_by_positions(string_tokens)
        else:
            return {0: string_tokens}, {0: list(range(len(string_tokens)))}

    def group_tokens_by_positions(
        self, string_tokens: List[str]
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[int]]]:
        tokens_group, positions_group = {}, {}
        for i, pos in enumerate(self.pivot_positions):
            if i == 0:
                positions_group[i] = [0, pos]
            else:
                positions_group[i] = [self.pivot_positions[i - 1], pos]
        positions_group[len(self.pivot_positions)] = [
            self.pivot_positions[-1],
            len(string_tokens),
        ]

        # modify the positions_group to include all the indexes and not just the start and end
        for i in range(len(positions_group)):
            positions_group[i] = list(
                range(positions_group[i][0], positions_group[i][1])
            )

        for i, group in positions_group.items():
            tokens_group[i] = [string_tokens[j] for j in group]
        return tokens_group, positions_group

    def group_tokens_by_pivot_tokens(
        self, string_tokens: List[str]
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[int]]]:
        tokens_group, positions_group = {}, {}
        current_group = 0
        start_pos = 0

        for i, token in enumerate(string_tokens):
            if isinstance(self.pivot_tokens, list) and token in self.pivot_tokens:
                positions_group[current_group] = [start_pos, i]
                tokens_group[current_group] = string_tokens[start_pos:i]
                current_group += 1
                start_pos = i + 1

        positions_group[current_group] = [start_pos, len(string_tokens)]
        tokens_group[current_group] = string_tokens[start_pos:]

        return tokens_group, positions_group

    def get_token_index(
        self,
        tokens: Union[
            List[Union[str, int, Tuple[int, int]]],
            List[str],
            List[int],
            List[Tuple[int, int]],
        ],
        string_tokens: List[str],
        return_type: Literal["list", "dict", "all"] = "list",
    ) -> Union[List[int], Dict, Tuple[List[int], Dict]]:
        """Get indices for specified tokens in the input sequence.

        Arguments:
            tokens: list of tokens to look up. Can be:
                - String token identifiers (see Supported Tokens below)
                - Integer indices (direct position access)
                - Tuple[int, int] for slices (start, end)
            string_tokens: Input token sequence to search in
            return_type: Output format:
                - "list": Returns [int, ...]
                - "dict": Returns {"token_type": [indices]}
                - "all": Returns ([indices], {token_type: [indices]})

        Returns:
            Based on return_type, provides token indices in specified format

        Token Types:
            1. Direct Access:
                - Integer index: Direct position (e.g. 5)
                - Slice tuple: Range of positions (e.g. (2,5))
                - Negative index: From end (e.g. -1)

            2. Named Positions:
                - "last": Last token
                - "last-2": Second to last
                - "last-4": Fourth to last

            3. Text Tokens:
                - "all-text": All text positions
                - "random-text": Single random text position
                - "random-text-N": N random text positions

            4. Image Tokens:
                - "all-image": All image positions
                - "last-image": Last image position
                - "end-image": Image end marker
                - "random-image": Random image position
                - "random-image-N": N random image positions

            5. Special:
                - "special": Model-specific tokens
                - "all": Complete sequence
                - "special-pixtral": Model-specific fixed positions
                - "inputs-partition-N": Nth token group
                - "random-inputs-partition-N": Random from group N

        Examples:
            # Direct index access
            >>> get_token_index([5], tokens)               # Single position
            [5]
            >>> get_token_index([(2,5)], tokens)          # Slice range
            [2,3,4]
            >>> get_token_index([-1], tokens)             # From end
            [9]

            # Named tokens
            >>> get_token_index(["last"], tokens)         # Last token
            [9]
            >>> get_token_index(["all-text"], tokens)     # All text
            [0,1,2,3]

            # Mixed usage
            >>> get_token_index(["last", (2,5)], tokens, return_type="dict")
            {"last": [9], "numeric": [2,3,4]}

            # Partition access
            >>> get_token_index(["inputs-partition-0"], tokens)  # First group
            [0,1,2]
            >>> get_token_index(["random-inputs-partition-0"], tokens) # Random from group
            [1]
        """
        # Convert single token to list for uniform processing
        # assert that we have a list of tokens
        if not isinstance(tokens, list):
            raise ValueError(
                "Tokens must be a list of strings or integers or tuples. Got {}".format(
                    type(tokens)
                )
            )

        # Check if all tokens are supported or not
        if not all(
            token in SUPPORTED_TOKENS
            or isinstance(token, int)
            or isinstance(token, tuple)
            or token.startswith("inputs-partition-")
            or token.startswith("random-inputs-partition-")
            or token.startswith("random-image")
            or token.startswith("random-text")
            for token in tokens
        ):
            raise ValueError(
                f"Unsupported token type into: {tokens}. Supported tokens are: {SUPPORTED_TOKENS} and inputs-partition-0, inputs-partition-1, etc or random-inputs-partition-0, random-inputs-partition-1, etc or integer indices or slices (tuple of start, end)"
            )

        # Check if pivot_positions is required but not provided
        if self.pivot_positions is None and any(
            isinstance(token, str)
            and (
                token.startswith("inputs-partition-")
                or token.startswith("random-inputs-partition-")
            )
            for token in tokens
        ):
            raise ValueError(
                "pivot_positions cannot be None when a group position token is requested"
            )

        token_indexes = self.categorize_tokens(string_tokens)
        tokens_positions, token_dict = self.get_tokens_positions(tokens, token_indexes)

        if return_type == "dict":
            return token_dict
        if return_type == "all":
            return tokens_positions, token_dict
        return tokens_positions

    def get_tokens_positions(
        self,
        tokens: List[Union[str, int, Tuple[int, int]]],
        token_indexes: Dict[str, List[int]],
    ) -> List[int]:
        tokens_positions = []
        position_dict = {
            k: v for k, v in token_indexes.items() if k.startswith("inputs-partition-")
        }
        random_position_dict = {
            f"random-{k}": random.sample(v, 1) for k, v in position_dict.items()
        }
        numeric_tokens = {}
        for token in tokens:
            if isinstance(token, int):
                # tokens_positions.append((token))
                # check if the absolute value of the token is less than the length of the string tokens
                if abs(token) >= len(token_indexes["all"]):
                    raise ValueError(f"Token {token} is out of range")
                numeric_tokens[token] = [token]
            elif isinstance(token, tuple):
                start, end = token
                # check if the start and end are within the range of the string tokens
                if start >= len(token_indexes["all"]) or end >= len(
                    token_indexes["all"]
                ):
                    raise ValueError(f"Token {token} is out of range")
                elif start > end and end > 0:
                    raise ValueError(
                        f"Token {token} is invalid. The start index must be less than the end index"
                    )
                if start > 0 and end < 0:
                    end = len(token_indexes["all"]) - end
                numeric_tokens[token] = list(range(start, end))
            elif token.startswith("random-inputs-partition-"):
                group, n = self.parse_random_group_token(token)
                random_position_dict[token] = random.sample(
                    position_dict[f"inputs-partition-{group}"], int(n)
                )
            elif token.startswith("random-image"):
                
                n = token.split("-")[-1]
                if n.isdigit():
                    n = int(n)
                else:
                    n = None
                random_position_dict[token] = random.sample(
                    token_indexes["image"], n if n else 1
                )
            elif token.startswith("random-text"):
                if token == "random-text":
                    random_position_dict[token] = random.sample(
                        token_indexes["text"], 1
                    )
                else:
                    n = token.split("-")[-1]
                    if n.isdigit():
                        n = int(n)
                    else:
                        n = None
                    random_position_dict[token] = random.sample(
                        token_indexes["text"], n if n else 1
                    )
            
        token_dict = self.get_token_dict(token_indexes, random_position_dict)
        # add the numeric indexe
        token_dict = {**token_dict, **numeric_tokens}

        for token in tokens:
            if token_dict[token] is not None:
                tokens_positions.append(tuple(token_dict[token]))  # type: ignore

        return tokens_positions, token_dict

    def parse_random_group_token(self, token: str) -> Tuple[str, int]:
        group_and_n = token.split("-")[2:]
        if len(group_and_n) > 1:
            group, n = group_and_n
        else:
            group = group_and_n[0]
            n = 1
        return group, int(n)

    def get_token_dict(
        self,
        token_indexes: Dict[str, List[int]],
        random_position_dict: Dict[str, List[int]] = {},
    ) -> Dict[str, Optional[List[int]]]:
        return {
            "last": [-1],
            "last-2": [-2],
            "last-4": [-4],
            "last-image": token_indexes["last_line_image"],
            "end-image": token_indexes["image_end"],
            "all-text": token_indexes["text"],
            "all": token_indexes["all"],
            "all-image": token_indexes["image"],
            "special": token_indexes["special"],
            "random-text": None
            if len(token_indexes["text"]) == 0
            else [random.choice(token_indexes["text"])],
            "random-image": None
            if len(token_indexes["image"]) == 0
            else [random.choice(token_indexes["image"])],
            "special-pixtral": [1052, 1051, 1038, 991, 1037, 1047],
            **{
                k: v
                for k, v in token_indexes.items()
                if k.startswith("inputs-partition-")
            },
            **random_position_dict,
        }
