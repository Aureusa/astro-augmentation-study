from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset

from optastra.data.sample import Sample
from task.gdz_5_schema import GZD5_QUESTIONS, GZD5Question


def _vote_count_columns(question: GZD5Question) -> list[str]:
    """Map a GZD5 question to its vote-count columns in the parquet file.

    The schema uses underscores while parquet columns use hyphens for
    question/answer names, e.g.

        featured_or_disk -> featured-or-disk
        more_than_4      -> more-than-4
    """
    q_col = question.name.replace("_", "-")

    answer_col_map = {
        "featured_or_disk": "featured-or-disk",
        "in_between": "in-between",
        "cigar_shaped": "cigar-shaped",
        "more_than_4": "more-than-4",
        "cant_tell": "cant-tell",
        "minor_disturbance": "minor-disturbance",
        "major_disturbance": "major-disturbance",
    }

    return [
        f"{q_col}_{answer_col_map.get(answer, answer)}"
        for answer in question.answers
    ]


class DECaLSDataset(Dataset):
    """DECaLS DR5 Galaxy Zoo dataset backed directly by ZIP archives.

    Expected layout:

        data_root/
            gz_decals_volunteers_5.parquet
            gz_decals_dr5_png_part1.zip
            gz_decals_dr5_png_part2.zip
            gz_decals_dr5_png_part3.zip
            gz_decals_dr5_png_part4.zip

    The parquet's png_loc values look like:

        dr5/J112/J112953.88-000427.4.png

    while the ZIP entries look like:

        J112/J112953.88-000427.4.png

    Therefore "dr5/" is removed before looking up an image.

    ZIP partitioning:

        J000-J100 -> part1
        J101-J120 -> part2
        J121-J150 -> part3
        J151-J235 -> part4

    ZIP files are opened lazily inside each DataLoader worker. This is
    important: do NOT open them in __init__, because Dataset instances
    are copied into worker processes.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        data_root: str | Path,
        transform: Callable[[Sample], Sample] | None = None,
        min_votes: int | None = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.questions = GZD5_QUESTIONS

        # ------------------------------------------------------------------
        # Load metadata once.
        # ------------------------------------------------------------------
        df = pd.read_parquet(parquet_path)

        # Remove rows without an image.
        df = df[df["png_loc"].notna()]

        # Optional filtering.
        if min_votes is not None:
            df = df[
                df["smooth-or-featured_total-votes"] >= min_votes
            ]

        df = df.reset_index(drop=True)

        # ------------------------------------------------------------------
        # Image paths.
        #
        # Parquet:
        #     dr5/J112/foo.png
        #
        # ZIP:
        #     J112/foo.png
        # ------------------------------------------------------------------
        self.paths = [
            path.removeprefix("dr5/")
            for path in df["png_loc"].tolist()
        ]

        # ------------------------------------------------------------------
        # Build target-column mapping.
        # ------------------------------------------------------------------
        self._question_columns: dict[str, list[str]] = {
            question.name: _vote_count_columns(question)
            for question in self.questions
        }

        # Check that every expected target column exists.
        missing = [
            column
            for columns in self._question_columns.values()
            for column in columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"DECaLSDataset: {len(missing)} expected vote-count "
                f"columns were not found in the parquet file. "
                f"Examples: {missing[:10]}"
            )

        # ------------------------------------------------------------------
        # Flatten all target columns into one contiguous matrix.
        #
        # Instead of doing this for every sample:
        #
        #     row = df.iloc[idx]
        #     row[cols].to_numpy()
        #     torch.tensor(...)
        #
        # we do the expensive pandas operation once here.
        # ------------------------------------------------------------------
        target_columns: list[str] = []
        self._target_slices: dict[str, slice] = {}

        offset = 0

        for question in self.questions:
            columns = self._question_columns[question.name]

            start = offset
            end = start + len(columns)

            self._target_slices[question.name] = slice(start, end)

            target_columns.extend(columns)
            offset = end

        self.targets = df[target_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        # ------------------------------------------------------------------
        # ZIP handles.
        #
        # These MUST be opened lazily, because each DataLoader worker is
        # a separate process and should have its own ZipFile objects.
        # ------------------------------------------------------------------
        self._zips: dict[int, zipfile.ZipFile] | None = None

    def __len__(self) -> int:
        return len(self.paths)

    # ----------------------------------------------------------------------
    # ZIP handling
    # ----------------------------------------------------------------------

    def _get_zips(self) -> dict[int, zipfile.ZipFile]:
        """Open ZIP archives lazily in the current worker process."""
        if self._zips is None:
            self._zips = {
                1: zipfile.ZipFile(
                    self.data_root / "gz_decals_dr5_png_part1.zip",
                    mode="r",
                ),
                2: zipfile.ZipFile(
                    self.data_root / "gz_decals_dr5_png_part2.zip",
                    mode="r",
                ),
                3: zipfile.ZipFile(
                    self.data_root / "gz_decals_dr5_png_part3.zip",
                    mode="r",
                ),
                4: zipfile.ZipFile(
                    self.data_root / "gz_decals_dr5_png_part4.zip",
                    mode="r",
                ),
            }

        return self._zips

    @staticmethod
    def _zip_number(path: str) -> int:
        """Return the ZIP archive containing `path`.

        Example:

            J112/J112953.88-000427.4.png
            -> 2
        """
        prefix = path[:4]

        if not prefix.startswith("J"):
            raise ValueError(
                f"Unexpected DECaLS image path: {path!r}"
            )

        try:
            number = int(prefix[1:])
        except ValueError as exc:
            raise ValueError(
                f"Invalid DECaLS image prefix: {prefix!r}"
            ) from exc

        if 0 <= number <= 100:
            return 1

        if 101 <= number <= 120:
            return 2

        if 121 <= number <= 150:
            return 3

        if 151 <= number <= 235:
            return 4

        raise KeyError(
            f"No ZIP archive configured for image prefix {prefix!r}"
        )

    # ----------------------------------------------------------------------
    # Image loading
    # ----------------------------------------------------------------------

    def _load_image(self, path: str) -> torch.Tensor:
        """Load one RGB PNG from the appropriate ZIP archive."""

        zips = self._get_zips()
        zip_number = self._zip_number(path)
        zf = zips[zip_number]

        try:
            with zf.open(path, mode="r") as f:
                png_bytes = f.read()
        except KeyError as exc:
            raise FileNotFoundError(
                f"Image {path!r} was not found in part{zip_number}.zip"
            ) from exc

        # Make a writable copy so torch.frombuffer() does not reference
        # immutable Python bytes.
        png_buffer = bytearray(png_bytes)

        image = torchvision.io.decode_image(
            torch.frombuffer(
                png_buffer,
                dtype=torch.uint8,
            ),
            mode=torchvision.io.ImageReadMode.RGB,
        )

        return image

    # ----------------------------------------------------------------------
    # Dataset access
    # ----------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Sample:
        # ------------------------------------------------------------------
        # Image
        # ------------------------------------------------------------------
        path = self.paths[idx]

        image = self._load_image(path)

        # ------------------------------------------------------------------
        # Targets
        # ------------------------------------------------------------------
        row = self.targets[idx]

        targets = {
            question.name: torch.from_numpy(
                row[self._target_slices[question.name]]
            )
            for question in self.questions
        }

        # ------------------------------------------------------------------
        # Sample
        # ------------------------------------------------------------------
        sample = Sample(
            image=image,
            target=targets,
        )

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------

    def __del__(self):
        """Close ZIP handles if this Dataset object owns any."""
        if self._zips is not None:
            for zf in self._zips.values():
                try:
                    zf.close()
                except Exception:
                    pass
