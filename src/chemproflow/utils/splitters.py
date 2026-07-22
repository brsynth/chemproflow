"""
Adapted from:
https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
"""
from collections import defaultdict
from typing import Generator, Iterator, Optional

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False
    
    Return: 
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


class Splitter(object):
    """Splitters split up Datasets into pieces for training/validation/testing.

    In machine learning applications, it's often necessary to split up a dataset
    into training/validation/test sets. Or to k-fold split a dataset (that is,
    divide into k equal subsets) for cross-validation. The `Splitter` class is
    an abstract superclass for all splitters that captures the common API across
    splitter classes.

    Note that `Splitter` is an abstract superclass. You won't want to
    instantiate this class directly. Rather you will want to use a concrete
    subclass for your application.
    """
    def k_fold_split(self, df: pd.DataFrame, k: int) -> Generator:
        """
        Scaffold/group-preserving k-fold split.

        Yields
        ------
        fold: int
        train_idx: list
            Indices for training.
        valid_idx: list
            Indices for validation for this fold.
        """
        if k < 2:
            raise ValueError("k must be at least 2.")
        if k > len(df):
            raise ValueError("k cannot be larger than the number of rows.")

        # Build scaffold groups using the subclass logic when possible.
        all_scaffolds = defaultdict(list)
        for ix, row in df.iterrows():
            scaffold = generate_scaffold(row["smiles"], include_chirality=True)
            all_scaffolds[scaffold].append(ix)

        scaffold_sets = [
            sorted(scaffold_set)
            for _, scaffold_set in sorted(
                all_scaffolds.items(),
                key=lambda x: (len(x[1]), x[1][0]),
                reverse=True,
            )
        ]

        # Greedy bin-packing: assign each scaffold group to the currently
        # smallest fold to balance fold sizes.
        folds = [[] for _ in range(k)]
        fold_sizes = [0 for _ in range(k)]

        for scaffold_set in scaffold_sets:
            smallest_fold = int(np.argmin(fold_sizes))
            folds[smallest_fold].extend(scaffold_set)
            fold_sizes[smallest_fold] += len(scaffold_set)

        all_indices = set(df.index)

        for fold in range(k):
            valid_idx = sorted(folds[fold])
            train_idx = sorted(all_indices - set(valid_idx))
            yield fold, train_idx, valid_idx

    def k_fold_split_stratified(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        k: int,
        random_state: int = 0,
    ) -> Generator:
        """
        Scaffold-preserving k-fold split with multi-label stratification.

        Each scaffold group is treated as an atomic unit. A group-level label
        vector (binary OR over member molecules) is used to stratify fold
        assignment via MultilabelStratifiedKFold, so rare labels are spread
        evenly across folds while scaffold integrity is maintained.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a "smiles" column. Its index labels must be valid
            integer positions into `labels` (i.e. the original df has a
            fresh 0-based RangeIndex and this is a .loc[] subset of it).
        labels : np.ndarray
            Full binary label matrix of shape [N_total, num_classes],
            indexable by df's index labels.
        k : int
            Number of folds.
        random_state : int
            Passed to MultilabelStratifiedKFold for reproducibility.

        Yields
        ------
        fold : int
        train_idx : list
        valid_idx : list
        """
        if k < 2:
            raise ValueError("k must be at least 2.")
        if k > len(df):
            raise ValueError("k cannot be larger than the number of rows.")

        # Build scaffold groups
        all_scaffolds = defaultdict(list)
        for ix, row in df.iterrows():
            scaffold = generate_scaffold(row["smiles"], include_chirality=True)
            all_scaffolds[scaffold].append(ix)

        scaffold_sets = [
            sorted(group)
            for _, group in sorted(
                all_scaffolds.items(),
                key=lambda x: (len(x[1]), x[1][0]),
                reverse=True,
            )
        ]

        num_groups = len(scaffold_sets)
        if k > num_groups:
            raise ValueError(
                f"k={k} exceeds the number of scaffold groups ({num_groups})."
            )

        # Group-level label matrix: binary OR over member molecules
        num_classes = labels.shape[1]
        group_labels = np.zeros((num_groups, num_classes), dtype=np.float32)
        for g_idx, group in enumerate(scaffold_sets):
            group_labels[g_idx] = labels[group].any(axis=0).astype(np.float32)

        # Stratified k-fold over scaffold groups
        group_indices = np.arange(num_groups).reshape(-1, 1)
        mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

        all_mol_indices = set(df.index)
        for fold, (_, val_group_pos) in enumerate(mskf.split(group_indices, group_labels)):
            valid_idx = sorted(idx for g in val_group_pos for idx in scaffold_sets[g])
            train_idx = sorted(all_mol_indices - set(valid_idx))
            yield fold, train_idx, valid_idx


class ScaffoldSplitter(Splitter):
    """
    Splits internal compounds into train/validation/test by scaffold.
    """
    def __init__(self):
        super(ScaffoldSplitter, self).__init__()
    
    def split(self,
        df: pd.DataFrame,
        frac_train: Optional[float] = None,
        frac_valid: Optional[float] = None,
        frac_test: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            df(pd.DataFrame): the dataset to split. Make sure each element in
                the dataset has key "smiles" which will be used to calculate the
                scaffold.
            frac_train(float): the fraction of data to be used for the train split.
            frac_valid(float): the fraction of data to be used for the valid split.
            frac_test(float): the fraction of data to be used for the test split.
            random_state(int): if set, randomizes the tie-break order among
                same-size scaffold groups so the resulting split varies across
                seeds. Groups are still packed largest-first to keep the split
                close to the requested fractions. If None, tie-breaking is
                deterministic (original behavior).
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = defaultdict(list)
        for ix, row in df.iterrows():
            scaffold = generate_scaffold(row["smiles"], include_chirality=True)
            all_scaffolds[scaffold].append(ix)

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        scaffold_items = list(all_scaffolds.items())

        if random_state is not None:
            # Shuffle first so the stable sort below breaks size ties randomly.
            np.random.RandomState(random_state).shuffle(scaffold_items)
            all_scaffold_sets = [
                scaffold_set for (_, scaffold_set) in sorted(
                    scaffold_items, key=lambda x: len(x[1]), reverse=True)
            ] # List[List[int]]
        else:
            all_scaffold_sets = [
                scaffold_set for (_, scaffold_set) in sorted(
                    scaffold_items, key=lambda x: (len(x[1]), x[1][0]), reverse=True)
            ] # List[List[int]]

        # get train, valid test indices
        train_cutoff = frac_train * len(df)
        valid_cutoff = (frac_train + frac_valid) * len(df)
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        return train_idx, valid_idx, test_idx

    def split_balanced_groups(
        self,
        df: pd.DataFrame,
        random_state: Optional[int] = None,
    ):
        """
        Split by scaffold into two halves, balancing the NUMBER of distinct
        scaffold groups (not the number of molecules) between train and test.

        `split()` packs scaffold groups by molecule-count fractions, so on a
        small dataset (e.g. one TC-ID's substrates) the single largest
        scaffold group can be dumped entirely on one side, leaving the other
        side with only one (or very few) distinct scaffolds. This method
        instead assigns each scaffold group, largest first, to whichever side
        currently holds fewer groups, guaranteeing the two sides differ by at
        most one scaffold group.

        Args:
            df(pd.DataFrame): the dataset to split; must have a "smiles" column.
            random_state(int): shuffles the tie-break order among same-size
                scaffold groups so the split varies across seeds. If None,
                tie-breaking is deterministic.

        Returns:
            train_idx, test_idx: lists of df index labels.
        """
        all_scaffolds = defaultdict(list)
        for ix, row in df.iterrows():
            scaffold = generate_scaffold(row["smiles"], include_chirality=True)
            all_scaffolds[scaffold].append(ix)

        scaffold_items = list(all_scaffolds.items())
        if len(scaffold_items) < 2:
            raise ValueError(
                "Need at least 2 distinct scaffolds to build a balanced "
                f"scaffold split; found {len(scaffold_items)}."
            )

        if random_state is not None:
            np.random.RandomState(random_state).shuffle(scaffold_items)
            all_scaffold_sets = [
                sorted(scaffold_set) for (_, scaffold_set) in sorted(
                    scaffold_items, key=lambda x: len(x[1]), reverse=True)
            ]
        else:
            all_scaffold_sets = [
                sorted(scaffold_set) for (_, scaffold_set) in sorted(
                    scaffold_items, key=lambda x: (len(x[1]), x[1][0]), reverse=True)
            ]

        train_idx, test_idx = [], []
        train_groups, test_groups = 0, 0
        for scaffold_set in all_scaffold_sets:
            if train_groups <= test_groups:
                train_idx.extend(scaffold_set)
                train_groups += 1
            else:
                test_idx.extend(scaffold_set)
                test_groups += 1

        return sorted(train_idx), sorted(test_idx)

    def split_stratified(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        frac_train: Optional[float] = None,
        frac_valid: Optional[float] = None,
        frac_test: Optional[float] = None,
        random_state: int = 0,
    ):
        """
        Scaffold-preserving single train/valid/test split with multi-label stratification.

        Each scaffold group is treated as an atomic unit. A group-level label
        vector (binary OR over member molecules) is used to stratify the split
        via MultilabelStratifiedShuffleSplit, so rare labels are spread evenly
        while scaffold integrity is maintained.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a "smiles" column. Its index labels must be valid
            integer positions into `labels`.
        labels : np.ndarray
            Full binary label matrix of shape [N_total, num_classes],
            indexable by df's index labels.
        frac_train, frac_valid, frac_test : float
            Fractions that must sum to 1.0.
        random_state : int
            Passed to MultilabelStratifiedShuffleSplit for reproducibility.

        Returns
        -------
        train_idx, valid_idx, test_idx : list
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

        # Build scaffold groups
        all_scaffolds = defaultdict(list)
        for ix, row in df.iterrows():
            scaffold = generate_scaffold(row["smiles"], include_chirality=True)
            all_scaffolds[scaffold].append(ix)

        scaffold_sets = [
            sorted(group)
            for _, group in sorted(
                all_scaffolds.items(),
                key=lambda x: (len(x[1]), x[1][0]),
                reverse=True,
            )
        ]

        num_groups = len(scaffold_sets)

        # Group-level label matrix: binary OR over member molecules
        num_classes = labels.shape[1]
        group_labels = np.zeros((num_groups, num_classes), dtype=np.float32)
        for g_idx, group in enumerate(scaffold_sets):
            group_labels[g_idx] = labels[group].any(axis=0).astype(np.float32)

        group_dummy = np.arange(num_groups).reshape(-1, 1)

        # Split scaffold groups into (train+valid) vs test
        msss_test = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=frac_test, random_state=random_state
        )
        trainval_pos, test_pos = next(msss_test.split(group_dummy, group_labels))

        # Split (train+valid) scaffold groups into train vs valid
        frac_valid_of_trainval = frac_valid / (frac_train + frac_valid)
        msss_valid = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=frac_valid_of_trainval, random_state=random_state
        )
        trainval_dummy = np.arange(len(trainval_pos)).reshape(-1, 1)
        train_sub, valid_sub = next(
            msss_valid.split(trainval_dummy, group_labels[trainval_pos])
        )

        train_idx = sorted(idx for g in trainval_pos[train_sub] for idx in scaffold_sets[g])
        valid_idx = sorted(idx for g in trainval_pos[valid_sub] for idx in scaffold_sets[g])
        test_idx = sorted(idx for g in test_pos for idx in scaffold_sets[g])

        return train_idx, valid_idx, test_idx