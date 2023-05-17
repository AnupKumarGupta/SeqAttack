import os
import json
import pandas as pd

from datasets import load_dataset

from .ner import NERDataset


class NERHuggingFaceDataset(NERDataset):
    """
        Dataset wrapper to load named entity recognition datasets
        from https://huggingface.co/datasets
    """
    def __init__(
            self,
            name: str,
            split: str,
            label_names: list,
            num_examples: int = None,
            labels_map: dict = None):
        """Loads a HuggingFace dataset, downloading it if needed.
        
            Parameters
            ----------

            name:            the dataset name
            split:           the dataset split (e.g. train, test)
            label_names:     list that maps prediction to labels (e.g. ["O", "B-ORG", ...])
            num_examples:    the maximum number of examples to be loaded
            labels_map:      a dictionary (e.g. {0: 2, 1: 3, ...}) that remaps the ground truth,
                            useful to align the model output and the dataset
        """
        if os.path.isfile(name):
            dataset = json.loads(open(name).read())

            dataset_tokens, dataset_ner_tags = zip(*dataset["samples"])
            dataset_tokens = [sample.split(" ") for sample in dataset_tokens]
        else:
            dataset = load_dataset(name, None, split=split)

            dataset_ner_tags = dataset["ner_tags"]
            dataset_tokens = dataset["tokens"]

        if labels_map:
            examples_ner_tags = [
                [labels_map[tag] for tag in tags]
                for tags in dataset_ner_tags
            ]
        else:
            examples_ner_tags = dataset_ner_tags

        dataset = list(zip(
            [" ".join(x) for x in dataset_tokens],
            examples_ner_tags
        ))

        self.dataset_split = split

        super().__init__(
            dataset,
            label_names,
            num_examples=num_examples,
            dataset_name=name
        )

    @staticmethod
    def from_config_file(path, num_examples=None):
        with open(path) as config_file:
            config = json.loads(config_file.read())

            labels_map = config["labels_map"]

            if labels_map is not None:
                # Convert keys to integers if a labels_map was provided
                labels_map = {int(k) : int(v) for k, v in labels_map.items()}

            return NERHuggingFaceDataset(
                name=config["name"],
                split=config["split"],
                label_names=config["labels"],
                num_examples=num_examples,
                labels_map=labels_map
            )

    @property
    def split(self):
        """The dataset split if any"""
        return self.dataset_split


class NERHuggingFaceDatasetForBioBERT(NERDataset):
    """
        Dataset wrapper to load named entity recognition datasets
        from https://huggingface.co/datasets
    """

    def __init__(
            self,
            name: str,
            dataset,
            split: str,
            label_names: list,
            num_examples: int = None,
            labels_map: dict = None):
        """Loads a HuggingFace dataset, downloading it if needed.

            Parameters
            ----------

            name:            the dataset name
            split:           the dataset split (e.g. train, test)
            label_names:     list that maps prediction to labels (e.g. ["O", "B-ORG", ...])
            num_examples:    the maximum number of examples to be loaded
            labels_map:      a dictionary (e.g. {0: 2, 1: 3, ...}) that remaps the ground truth,
                            useful to align the model output and the dataset
        """
        # labels_path = "/Users/anupkumargupta/PycharmProjects/SeqAttack/datasets/NCBI-disease/labels.txt"
        # inputs_path = "/Users/anupkumargupta/PycharmProjects/SeqAttack/datasets/NCBI-disease/test.tsv"
        #
        # labels_dataframe = pd.read_csv(labels_path, header=None)
        # label_names = labels_dataframe[0].tolist()
        #
        # data_values = []
        # label_values = []
        # dataset = []
        #
        # dataset_dataframe = pd.read_csv(inputs_path, sep='\t', header=None, skip_blank_lines=False,
        #                                 keep_default_na=False, na_values="")
        #
        # for index, row in dataset_dataframe.iterrows():
        #     if row.isnull().any():
        #         string_data = " ".join(data_values)
        #         dataset.append((string_data, label_values))
        #         data_values = []
        #         label_values = []
        #     else:
        #         data_values.append(row[0])
        #         label_values.append(row[1])
        #
        # dataset_split = "test"
        # split = dataset_split
        #
        # dataset_name = "test"
        # name = dataset_name

        # if os.path.isfile(name):
        #     dataset = json.loads(open(name).read())
        #
        #     dataset_tokens, dataset_ner_tags = zip(*dataset["samples"])
        #     dataset_tokens = [sample.split(" ") for sample in dataset_tokens]
        # else:
        #     dataset = load_dataset(name, None, split=split)
        #
        #     dataset_ner_tags = dataset["ner_tags"]
        #     dataset_tokens = dataset["tokens"]
        #
        # if labels_map:
        #     examples_ner_tags = [
        #         [labels_map[tag] for tag in tags]
        #         for tags in dataset_ner_tags
        #     ]
        # else:
        #     examples_ner_tags = dataset_ner_tags
        #
        # dataset = list(zip(
        #     [" ".join(x) for x in dataset_tokens],
        #     examples_ner_tags
        # ))
        #
        # self.dataset_split = split

        self.dataset_split = split
        super().__init__(
            dataset,
            label_names,
            num_examples=num_examples,
            dataset_name=name
        )

    @staticmethod
    def from_tsv_file(path, num_examples=None):
        # dir_path = "/Users/anupkumargupta/PycharmProjects/SeqAttack"
        # labels_path = os.path.join(dir_path, path, "labels.txt")
        # inputs_path = os.path.join(dir_path, path, "test.tsv")

        labels_path = os.path.join(path, "labels.txt")
        inputs_path = os.path.join(path, "test.tsv")
        labels_dataframe = pd.read_csv(labels_path, header=None)
        label_names = labels_dataframe[0].tolist()

        label_map = {_: _ for _ in range(len(label_names))}
        label_to_idx = {label: idx for idx, label in enumerate(label_names)}

        data_values = []
        label_values = []
        dataset = []

        dataset_dataframe = pd.read_csv(inputs_path, sep='\t', header=None, skip_blank_lines=False,
                                        keep_default_na=False, na_values="")

        for index, row in dataset_dataframe.iterrows():
            if row.isnull().any():
                string_data = " ".join(data_values)
                dataset.append((string_data, label_values))
                data_values = []
                label_values = []
            else:
                data_values.append(row[0])
                label_values.append(label_to_idx[row[1]])

        dataset_name = path.split("/")[-1].lower()

        return NERHuggingFaceDatasetForBioBERT(
            name=dataset_name,
            dataset=dataset,
            split="test",
            label_names=label_names,
            num_examples=num_examples,
            labels_map=label_map
        )

    @staticmethod
    def from_config_file(path, num_examples=None):
        with open(path) as config_file:
            config = json.loads(config_file.read())

            labels_map = config["labels_map"]

            if labels_map is not None:
                # Convert keys to integers if a labels_map was provided
                labels_map = {int(k): int(v) for k, v in labels_map.items()}

            return NERHuggingFaceDatasetForBioBERT(
                name=config["name"],
                split=config["split"],
                label_names=config["labels"],
                num_examples=num_examples,
                labels_map=labels_map
            )

    @property
    def split(self):
        """The dataset split if any"""
        return self.dataset_split
