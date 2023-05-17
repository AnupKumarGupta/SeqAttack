import os.path

import click
import random

from typing import Dict, List
from seqattack.models import NERModelWrapper
from seqattack.datasets import NERHuggingFaceDatasetForBioBERT, NERHuggingFaceDataset
from seqattack.utils.attack_runner import AttackRunner
from seqattack.goal_functions import get_goal_function
from seqattack.utils import postprocess_ner_output

from seqattack.attacks import (
    NERCLARE,
    BertAttackNER,
    NERBAEGarg2019,
    NERTextFoolerJin2019,
)
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer
)

attack_mode = True


def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
            labels = [i + '-bio' if i != 'O' else 'O' for i in labels]
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        # return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        return ["O", "B-bio", "I-bio"]


@click.group()
@click.option("--model-name", default="dslim/bert-base-NER")
@click.option("--output-path", type=str)
@click.option("--random-seed", default=567)
@click.option("--num-examples", default=256)
@click.option("--max-entities-mispredicted", default=0.8)
@click.option("--cache/--no-cache", default=False)
@click.option("--goal-function", default="untargeted")
@click.option("--max-queries", default=512)
@click.option("--attack-timeout", default=60)
@click.option("--dataset-config", default=None, required=True)
@click.pass_context
def attack(
        ctx,
        model_name,
        output_path,
        random_seed,
        num_examples,
        max_entities_mispredicted,
        cache,
        goal_function,
        max_queries,
        attack_timeout,
        dataset_config):
    random.seed(random_seed)

    goal_function_cls = get_goal_function(goal_function)

    if attack_mode:
        # model_name_or_path = os.path.join("/Users/anupkumargupta/PycharmProjects/SeqAttack", model_name)
        model_name_or_path = model_name
    else:
        model_name_or_path = model_name

    if attack_mode:
        dataset = NERHuggingFaceDatasetForBioBERT.from_tsv_file(
            dataset_config
        )
    else:
        dataset = NERHuggingFaceDataset.from_config_file(
            dataset_config,
            num_examples=num_examples
        )

    # Load model and tokenizer
    if attack_mode:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )

        labels_path = "datasets/NCBI-disease/labels.txt" # /Users/anupkumargupta/PycharmProjects/SeqAttack/
        labels = get_labels(labels_path)
        label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
        num_labels = len(labels)
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label=label_map,
            label2id={label: i for i, label in enumerate(labels)},
            cache_dir=None,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            config=config
        )

        model = NERModelWrapper(model=model, tokenizer=tokenizer, name=model_name,
                                postprocess_func=postprocess_ner_output)
    else:
        tokenizer, model = NERModelWrapper.load_huggingface_model(model_name)

    ctx.ensure_object(dict)
    ctx.obj["attack_args"] = {
        "random_seed": random_seed,
        "model": model,
        "model_name": model_name_or_path,
        "tokenizer": tokenizer,
        "goal_function_class": goal_function_cls,
        "use_cache": cache,
        "query_budget": max_queries,
        "dataset": dataset,
        "max_entities_mispredicted": max_entities_mispredicted,
        "output_path": output_path,
        "attack_timeout": attack_timeout,
        "num_examples": num_examples
    }


@attack.command()
@click.option("--max-words-perturbed", default=0.4)
@click.option("--max-candidates", default=48)
@click.pass_context
def bert_attack(ctx, max_words_perturbed, max_candidates):
    bert_attack_args = {
        "recipe": "bert-attack",
        "max_perturbed_percent": max_words_perturbed,
        "max_candidates": max_candidates,
        "additional_constraints": BertAttackNER.get_ner_constraints(ctx.obj["attack_args"]["model_name"])
    }

    ctx.obj["attack_args"] = {**ctx.obj["attack_args"], **bert_attack_args}
    ctx.obj["attack_args"]["recipe_metadata"] = bert_attack_args

    attack = BertAttackNER.build(**ctx.obj["attack_args"])

    AttackRunner(
        attack=attack,
        dataset=ctx.obj["attack_args"]["dataset"],
        output_filename=ctx.obj["attack_args"]["output_path"],
        attack_args=ctx.obj["attack_args"]
    ).run()


@attack.command()
@click.option("--max-candidates", default=48)
@click.pass_context
def clare(ctx, max_candidates):
    clare_attack_args = {
        "recipe": "clare",
        "max_candidates": max_candidates,
        "additional_constraints": NERCLARE.get_ner_constraints(ctx.obj["attack_args"]["model_name"])
    }

    ctx.obj["attack_args"] = {**ctx.obj["attack_args"], **clare_attack_args}
    ctx.obj["attack_args"]["recipe_metadata"] = clare_attack_args

    attack = NERCLARE.build(**ctx.obj["attack_args"])

    AttackRunner(
        attack=attack,
        dataset=ctx.obj["attack_args"]["dataset"],
        output_filename=ctx.obj["attack_args"]["output_path"],
        attack_args=ctx.obj["attack_args"]
    ).run()


@attack.command()
@click.option("--max-edit-distance", default=50)
@click.option("--preserve-named-entities/--no-preserve-named-entities", default=True)
@click.pass_context
def deepwordbug(ctx, max_edit_distance, preserve_named_entities):
    deepwordbug_args = {
        "recipe": "deepwordbug",
        "max_edit_distance": max_edit_distance,
        "additional_constraints": NERDeepWordBugGao2018.get_ner_constraints(
            ctx.obj["attack_args"]["model_name"],
            **{"preserve_named_entities": preserve_named_entities}
        )
    }

    ctx.obj["attack_args"] = {**ctx.obj["attack_args"], **deepwordbug_args}
    ctx.obj["attack_args"]["recipe_metadata"] = deepwordbug_args

    attack = NERDeepWordBugGao2018.build(**ctx.obj["attack_args"])

    AttackRunner(
        attack=attack,
        dataset=ctx.obj["attack_args"]["dataset"],
        output_filename=ctx.obj["attack_args"]["output_path"],
        attack_args=ctx.obj["attack_args"]
    ).run()


@attack.command()
@click.pass_context
def scpn(ctx):
    scpn_attack_args = {
        "recipe": "scpn",
        "additional_constraints": []
    }

    ctx.obj["attack_args"] = {**ctx.obj["attack_args"], **scpn_attack_args}
    ctx.obj["attack_args"]["recipe_metadata"] = scpn_attack_args

    attack = NERSCPNParaphrase.build(**ctx.obj["attack_args"])

    AttackRunner(
        attack=attack,
        dataset=ctx.obj["attack_args"]["dataset"],
        output_filename=ctx.obj["attack_args"]["output_path"],
        attack_args=ctx.obj["attack_args"]
    ).run()


@attack.command()
@click.option("--max-candidates", default=50)
@click.pass_context
def textfooler(ctx, max_candidates):
    textfooler_attack_args = {
        "recipe": "textfooler",
        "max_candidates": max_candidates,
        "additional_constraints": NERTextFoolerJin2019.get_ner_constraints(ctx.obj["attack_args"]["model_name"])
    }

    ctx.obj["attack_args"] = {**ctx.obj["attack_args"], **textfooler_attack_args}
    ctx.obj["attack_args"]["recipe_metadata"] = textfooler_attack_args

    attack = NERTextFoolerJin2019.build(**ctx.obj["attack_args"])

    AttackRunner(
        attack=attack,
        dataset=ctx.obj["attack_args"]["dataset"],
        output_filename=ctx.obj["attack_args"]["output_path"],
        attack_args=ctx.obj["attack_args"]
    ).run()


@attack.command()
@click.option("--max-candidates", default=50)
@click.pass_context
def bae(ctx, max_candidates):
    bae_attack_args = {
        "recipe": "bae",
        "max_candidates": max_candidates,
        "additional_constraints": NERBAEGarg2019.get_ner_constraints(ctx.obj["attack_args"]["model_name"])
    }

    ctx.obj["attack_args"] = {**ctx.obj["attack_args"], **bae_attack_args}
    ctx.obj["attack_args"]["recipe_metadata"] = bae_attack_args

    attack = NERBAEGarg2019.build(**ctx.obj["attack_args"])

    AttackRunner(
        attack=attack,
        dataset=ctx.obj["attack_args"]["dataset"],
        output_filename=ctx.obj["attack_args"]["output_path"],
        attack_args=ctx.obj["attack_args"]
    ).run()


@attack.command()
@click.pass_context
def morpheus(ctx):
    morpheus_args = {
        "recipe": "morpheus",
        "additional_constraints": MorpheusTan2020NER.get_ner_constraints(
            ctx.obj["attack_args"]["model_name"]
        )
    }

    ctx.obj["attack_args"] = {**ctx.obj["attack_args"], **morpheus_args}
    ctx.obj["attack_args"]["recipe_metadata"] = morpheus_args

    attack = MorpheusTan2020NER.build(**ctx.obj["attack_args"])

    AttackRunner(
        attack=attack,
        dataset=ctx.obj["attack_args"]["dataset"],
        output_filename=ctx.obj["attack_args"]["output_path"],
        attack_args=ctx.obj["attack_args"]
    ).run()
