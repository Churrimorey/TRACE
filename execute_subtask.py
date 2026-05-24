import argparse
import copy
import inspect
import json
import logging
import os
import random
import time

import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datetime import datetime
from typing import Any, Optional, Literal
from dataclasses import field
from pydantic import BaseModel

import evolution_utils as evolution_actions
from data_utils import TemplateSample, Memory
from evaluate_utils import evaluate_agentharm, evaluate_advcua, evaluate_practical

from action_markov import ActionMarkovQ, ACTIONS
from config import ensure_parent_dir


log_filename = f"logs/subtask_execution/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logger = logging.getLogger("execute_subtask")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ensure_parent_dir(log_filename)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)




def _execute_action_by_name(
    template_sample: TemplateSample,
    action_name: str,
    component_pool: dict
) -> None:
    action_fn = getattr(evolution_actions, action_name, None)
    if action_fn is None or not callable(action_fn):
        raise ValueError(f"Unknown or non-callable action: {action_name}")

    signature = inspect.signature(action_fn)
    available_args = {
        "template_sample": template_sample,
        "component_pool": component_pool,
    }

    call_kwargs: dict[str, Any] = {}
    for param_name, param in signature.parameters.items():
        if param_name in available_args:
            call_kwargs[param_name] = available_args[param_name]
        elif param.default is inspect.Parameter.empty:
            raise TypeError(
                f"Action `{action_name}` requires unsupported parameter `{param_name}`"
            )

    action_fn(**call_kwargs)



def _initialize_template_population(
    args: argparse.Namespace,
    step_datum: dict[str, Any],
    num_candidate_per_iter: int,
    # tool_file_path: str,
    unique_templates: Optional[set[str]] = None,
    **kwargs
) -> list[TemplateSample]:

    roles = COMPONENT_POOL["role"]
    environments = COMPONENT_POOL["environment"]
    directives = COMPONENT_POOL["directive"]
    tips_pool = COMPONENT_POOL["tips"]

    if not tips_pool:
        raise ValueError("tips pool is empty")

    population: list[TemplateSample] = []
    while len(population) < num_candidate_per_iter:
        num_tip = min(len(tips_pool), random.randint(args.min_num_tips, args.max_num_tips))
        # logger.debug(f"step_datum: {step_datum}")
        sample = TemplateSample(
            # tool_file_path=tool_file_path,
            datum=copy.deepcopy(step_datum),
            role=random.choice(roles),
            environment=random.choice(environments),
            directive=random.choice(directives),
            tips=random.sample(tips_pool, num_tip),
        )
        sample.source.append({"init": 0.0})
        sample.update_template()

        if sample.template in unique_templates:
            logger.info("Duplicate template found during initialization, skipping...")
            continue
        

        try:
            sample.evaluate(args.target_model_name, args.dataset, **kwargs)
            population.append(sample)
            unique_templates.add(sample.template)
            # logger.debug(f"prompt in _initialize_template_population: {step_datum['prompt']}")
            # logger.debug(f"template in _initialize_template_population: {sample.template}")
            # logger.debug(f"jailbroken_prompt in _initialize_template_population: {sample.datum['jailbroken_prompt']}")
            # logger.debug(f"tool_call_chains in _initialize_template_population: {sample.tool_call_chains}")
            # logger.debug(f"score in _initialize_template_population: {sample.score}, refusal: {sample.refusal}")
            # if sample.score - sample.refusal == 1.0:
            if sample.score - sample.refusal >= 0.95:
                logger.info("Found a perfect template during initialization, early stopping...")
                return population
        except Exception as e:
            logger.error(f"Error evaluating template sample during initialization: {e}")

    return population


def _initialize_template_population_with_fixed_templates(
    args: argparse.Namespace,
    step_datum: dict[str, Any],
    template_samples: list[TemplateSample],
    # tool_file_path: str,
    unique_templates: Optional[set[str]] = None,
    **kwargs
) -> list[TemplateSample]:

    population: list[TemplateSample] = []
    for template_sample in template_samples:
        # logger.debug(f"step_datum: {step_datum}")
        sample = TemplateSample(
            # tool_file_path=tool_file_path,
            datum=copy.deepcopy(step_datum),
            role=template_sample.role,
            environment=template_sample.environment,
            directive=template_sample.directive,
            tips=template_sample.tips,
        )
        sample.source.append({"init_from_memory": 0.0})
        sample.update_template()

        if sample.template in unique_templates:
            logger.info("Duplicate template found during initialization, skipping...")
            continue

        try:
            sample.evaluate(args.target_model_name, args.dataset, **kwargs)
            population.append(sample)
            unique_templates.add(sample.template)
            # logger.debug(f"prompt in _initialize_template_population: {step_datum['prompt']}")
            # logger.debug(f"template in _initialize_template_population: {sample.template}")
            # logger.debug(f"jailbroken_prompt in _initialize_template_population: {sample.datum['jailbroken_prompt']}")
            # logger.debug(f"tool_call_chains in _initialize_template_population: {sample.tool_call_chains}")
            # logger.debug(f"score in _initialize_template_population: {sample.score}, refusal: {sample.refusal}")
            # if sample.score - sample.refusal == 1.0:
            if sample.score - sample.refusal >= 0.95:
                logger.info("Found a perfect template during initialization from memory, early stopping...")
                return population
        except Exception as e:
            logger.error(f"Error evaluating template sample during initialization: {e}")

    return population


def save_intermediate_results(datum: dict[str, Any], output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_path = f"{output_dir}/{datum['id']}_{datum['step_id']}.jsonl"
    logger.info(f"saving intermediate result for datum id {datum['id']} step id {datum['step_id']} to {output_file_path}")
    with open(output_file_path, "a") as f:
        f.write(json.dumps(datum, ensure_ascii=False) + "\n")


def save_final_results(datum: dict[str, Any], output_dir: str):
    # "iter" "messages" "tool_call_chains" "score" "refusal" "time"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_path = f"{output_dir}/{datum['id']}.jsonl"
    with open(output_file_path, "a") as f:
        f.write(json.dumps(datum, ensure_ascii=False) + "\n")


def evolution_optimization_for_harmful_step(
    args: argparse.Namespace,
    harmful_step: dict[str, Any],
    **kwargs
) -> TemplateSample:
    unique_templates = set()
    early_stop = False

    start = time.time()
    if os.path.exists(f"{args.intermediate_result_dir}/{harmful_step['id']}_{harmful_step['step_id']}.jsonl"):
        current_population = []
        with open(f"{args.intermediate_result_dir}/{harmful_step['id']}_{harmful_step['step_id']}.jsonl", "r") as f:
            intermediate_results = []
            for line in f:
                intermediate_result = json.loads(line.strip())
                intermediate_results.append(intermediate_result)

            intermediate_results = intermediate_results[-args.num_candidate_per_iter:]
            
            unique_templates = set(intermediate_results[0]["unique_templates"])
            current_iter = intermediate_results[0]["iter"]
            for intermediate_result in intermediate_results:
                assert intermediate_result["iter"] == current_iter, logger.error("intermediate results have different iter, cannot be used for initialization")
                current_population.append(
                    TemplateSample(
                        datum=intermediate_result["datum"],
                        role=intermediate_result["role"],
                        environment=intermediate_result["environment"],
                        directive=intermediate_result["directive"],
                        tips=intermediate_result["tips"],
                        template=intermediate_result["template"],
                        messages=intermediate_result["messages"],
                        tool_call_chains=intermediate_result["tool_call_chains"],
                        score=intermediate_result["score"],
                        refusal=intermediate_result["refusal"],
                        source=intermediate_result["source"],
                        updated=True
                    )
                )

            start_iter = current_iter

    elif args.use_memory:
        template_samples_from_memory = MEMORY.query(
            harmful_step["prompt"],
            min(args.num_candidate_per_iter // 2, args.memory_top_k)
        )

        guided_current_population = _initialize_template_population_with_fixed_templates(
            args,
            harmful_step,
            template_samples_from_memory,
            # tool_file_path=tool_file_path,
            unique_templates=unique_templates,
            **kwargs
        )

        #  we put memory retrieval after checking intermediate results and before initialization of random templates
        # Memory retrieval directly succeed, no need for further optimization.
        # if len(guided_current_population) > 0 and guided_current_population[-1].score - guided_current_population[-1].refusal == 1.0:
        if len(guided_current_population) > 0 and guided_current_population[-1].score - guided_current_population[-1].refusal >= 0.95:
            logger.info("Found a perfect template from memory, early stopping...")
            end = time.time()

            harmful_step["iter"] = 0
            harmful_step["messages"] = guided_current_population[-1].messages
            harmful_step["tool_call_chains"] = guided_current_population[-1].tool_call_chains
            harmful_step["score"] = guided_current_population[-1].score
            harmful_step["refusal"] = guided_current_population[-1].refusal
            harmful_step["time"] = end - start
            harmful_step["source"] = guided_current_population[-1].source
            save_final_results(harmful_step, args.final_result_dir)

            return guided_current_population[-1]
        
        current_population = guided_current_population + _initialize_template_population(
            args,
            harmful_step,
            num_candidate_per_iter=args.num_candidate_per_iter - len(guided_current_population),
            # tool_file_path=tool_file_path,
            unique_templates=unique_templates,
            **kwargs
        )

        start_iter = 0
    else:
        current_population = _initialize_template_population(
            args,
            harmful_step,
            num_candidate_per_iter=args.num_candidate_per_iter,
            # tool_file_path=tool_file_path,
            unique_templates=unique_templates,
            **kwargs
        )
        start_iter = 0

    best_template_sample = None

    for i in range(start_iter, args.num_iters):
        current_population = sorted(current_population, key=lambda x: (x.score-x.refusal), reverse=True)
        if i > start_iter:
            if args.use_decision_matrix:
                for template_sample in current_population:
                    last_action_performantce = template_sample.source[-2]
                    new_action_performantce = template_sample.source[-1]
                    logger.debug("decision matrix update")
                    last_action = list(last_action_performantce.keys())[0]
                    if last_action == "init_from_memory":
                        last_action = "init"
                    DECISION_MATRIX.update(
                        last_action=last_action,
                        chosen_next_action=list(new_action_performantce.keys())[0],
                        reward=list(new_action_performantce.values())[0] - list(last_action_performantce.values())[0]
                    )
            if best_template_sample.score - best_template_sample.refusal < current_population[0].score - current_population[0].refusal:
                best_template_sample = copy.deepcopy(current_population[0])
        else:
            best_template_sample = copy.deepcopy(current_population[0])
        for template_sample in current_population:
            save_intermediate_results(
                datum={
                    "id": harmful_step["id"],
                    "step_id": harmful_step["step_id"],
                    "iter": i,
                    "datum": harmful_step,
                    "role": template_sample.role,
                    "environment": template_sample.environment,
                    "directive": template_sample.directive,
                    "tips": template_sample.tips,
                    "template": template_sample.template,
                    "messages": template_sample.messages,
                    "tool_call_chains": template_sample.tool_call_chains,
                    "score": template_sample.score,
                    "refusal": template_sample.refusal,
                    "source": template_sample.source,
                    "unique_templates": list(unique_templates)
                }, 
                output_dir=args.intermediate_result_dir
            )
            logger.debug({
                "role": template_sample.role,
                "environment": template_sample.environment,
                "directive": template_sample.directive,
                "tips": template_sample.tips,
                "template": template_sample.template,
                "source": template_sample.source,
                "score": template_sample.score,
                "refusal": template_sample.refusal
            })
        logger.info(f"iter {i} selected population scores {[template_sample.score - template_sample.refusal for template_sample in current_population]}")
        # if current_population[0].score - current_population[0].refusal == 1.0:
        if current_population[0].score - current_population[0].refusal >= 0.95:
            logger.info(f"early stop at iter {i} with score {current_population[0].score - current_population[0].refusal}")
            early_stop = True
            if args.use_memory:
                if not current_population[0].role in COMPONENT_POOL["role"]:
                    logger.info(f"adding new role in successful template to component pool")
                    COMPONENT_POOL["role"].append(current_population[0].role)
                if not current_population[0].environment in COMPONENT_POOL["environment"]:
                    logger.info(f"adding new environment in successful template to component pool")
                    COMPONENT_POOL["environment"].append(current_population[0].environment)
                if not current_population[0].directive in COMPONENT_POOL["directive"]:
                    logger.info(f"adding new directive in successful template to component pool")
                    COMPONENT_POOL["directive"].append(current_population[0].directive)
                for tip in current_population[0].tips:
                    if not tip in COMPONENT_POOL["tips"]:
                        logger.info(f"adding new tip in successful template to component pool")
                        COMPONENT_POOL["tips"].append(tip)
                logger.info(f"adding successful template to memory")
                MEMORY.add(
                    copy.deepcopy(current_population[0]),
                    prompt=harmful_step["prompt"],
                )
            break

        random.shuffle(current_population)

        for template_sample in current_population:
            try:
                if args.use_decision_matrix:
                    # if i > start_iter:
                    logger.debug("decision matrix select_next_action")
                    last_action = list(template_sample.source[-1].keys())[0]
                    if last_action == "init_from_memory":
                        last_action = "init"
                    next_action = DECISION_MATRIX.select_next_action(
                        last_action
                    )
                    logger.info(f"selected action {next_action} based on decision matrix")
                    # else: # initial vector
                    #     logger.debug("decision matrix select_initial_action")
                    #     next_action = DECISION_MATRIX.select_initial_action()
                else:
                    next_action = random.choice(ACTIONS)

                _execute_action_by_name(
                    template_sample=template_sample,
                    action_name=next_action,
                    component_pool=COMPONENT_POOL,
                )
                template_sample.source.append({next_action: 0.0})
                template_sample.clear()
                assert not template_sample.updated, logger.error("template already updated before update_template")
                template_sample.update_template()
                template_sample.evaluate(args.target_model_name, args.dataset, **kwargs)
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                raise e
        logger.info(f"after evaluation: {len(current_population)}")
    
    
    if not early_stop:
        current_population = sorted(current_population, key=lambda x: (x.score-x.refusal), reverse=True)
        logger.info(f"iter {i} selected population scores {[template_sample.score - template_sample.refusal for template_sample in current_population]}")
        for template_sample in current_population:
            save_intermediate_results(
                datum={
                    "id": harmful_step["id"],
                    "step_id": harmful_step["step_id"],
                    "iter": i,
                    "datum": harmful_step,
                    "role": template_sample.role,
                    "environment": template_sample.environment,
                    "directive": template_sample.directive,
                    "tips": template_sample.tips,
                    "template": template_sample.template,
                    "messages": template_sample.messages,
                    "tool_call_chains": template_sample.tool_call_chains,
                    "score": template_sample.score,
                    "refusal": template_sample.refusal,
                    "source": template_sample.source,
                    "unique_templates": list(unique_templates)
                }, 
                output_dir=args.intermediate_result_dir
            )
        if best_template_sample.score - best_template_sample.refusal < current_population[0].score - current_population[0].refusal:
            best_template_sample = copy.deepcopy(current_population[0])
        if args.use_decision_matrix:
            for template_sample in current_population:
                last_action_performantce = template_sample.source[-2]
                new_action_performantce = template_sample.source[-1]
                logger.debug("not early_stop decision matrix update")
                DECISION_MATRIX.update(
                    last_action=list(last_action_performantce.keys())[0],
                    chosen_next_action=list(new_action_performantce.keys())[0],
                    reward=list(new_action_performantce.values())[0] - list(last_action_performantce.values())[0]
                )
    end = time.time()

    harmful_step["iter"] = i + 1
    harmful_step["messages"] = best_template_sample.messages
    harmful_step["tool_call_chains"] = best_template_sample.tool_call_chains
    harmful_step["score"] = best_template_sample.score
    harmful_step["refusal"] = best_template_sample.refusal
    harmful_step["time"] = end - start
    harmful_step["source"] = best_template_sample.source
    save_final_results(harmful_step, args.final_result_dir)

    return best_template_sample



def execute_safe_step(datum: dict[str, Any], model_name: str, dataset: str, final_result_dir:str, **kwargs) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    start = time.time()
    try:
        if dataset == "agentharm":
            messages, tool_call_chains, score, refusal = evaluate_agentharm(datum, model_name, field="prompt")
        elif dataset == "advcua":
            messages, tool_call_chains, score, refusal = evaluate_advcua(datum, model_name, field="prompt", last_subtask=kwargs.get("last_subtask", False))
        elif dataset == "practical":
            messages, tool_call_chains, score, refusal = evaluate_practical(datum, model_name, field="prompt")
    except Exception as e:
        logger.error(f"Error during safe step execution: {e}")
        raise e
    end = time.time()
    # if score - refusal == 1.0:
    if score - refusal >= 0.95:
        logger.info(f"safe step executed successfully with score 1.0")
        datum["iter"] = 0
        datum["messages"] = messages
        datum["tool_call_chains"] = tool_call_chains
        datum["score"] = score
        datum["refusal"] = refusal
        datum["time"] = end - start
        save_final_results(datum, final_result_dir)
        return messages, tool_call_chains
    return None, None


def execute_steps(
    args: argparse.Namespace,
    datum: dict[str, Any],
    intermediate_results: Optional[list[dict[str, Any]]] = None,
    # tool_file_path: str = DEFAULT_TOOL_FILE_PATH,
) -> list[dict[str, Any]]:
    sub_prompts = datum["sub_prompts"]
    harmful_scores = datum["harmful_scores"]
    assert len(sub_prompts) == len(harmful_scores)
    if intermediate_results is not None:
        history_conv: list[dict[str, str]] = [message for result in intermediate_results for message in result["messages"]]
        logger.debug(f"intermediate history_conv: {history_conv}")
        tool_call_chains_history : list[dict[str, str]] = [tool_call_chain for result in intermediate_results for tool_call_chain in result["tool_call_chains"]]
        logger.debug(f"intermediate tool_call_chains_history: {tool_call_chains_history}")
    else:
        history_conv: list[dict[str, str]] = []
        tool_call_chains_history: list[dict[str, str]] = []
    

    for idx, (sub_prompt, harmful_score) in enumerate(zip(sub_prompts, harmful_scores)):
        if intermediate_results is not None and idx < len(intermediate_results):
            continue
        if harmful_score <= args.threshold:
            if args.dataset == "agentharm":
                messages, tool_call_chains = execute_safe_step(
                    datum={"id": datum["id"], "prompt": sub_prompt, "history": tool_call_chains_history if len(tool_call_chains_history) > 0 else None},
                    model_name=args.target_model_name,
                    dataset=args.dataset,
                    final_result_dir=args.final_result_dir
                )
            elif args.dataset == "advcua":
                logger.debug(f"last_subtask: {(idx==len(sub_prompts)-1)}")
                temp_datum = copy.deepcopy(datum)
                temp_datum.pop("sub_prompts")
                temp_datum.pop("harmful_scores")
                temp_datum["prompt"] = sub_prompt
                temp_datum["history"] = tool_call_chains_history if len(tool_call_chains_history) > 0 else None
                messages, tool_call_chains = execute_safe_step(
                    datum=temp_datum,
                    # datum={"id": datum["id"], "prompt": sub_prompt, "history": tool_call_chains_history if len(tool_call_chains_history) > 0 else None},
                    model_name=args.target_model_name,
                    dataset=args.dataset,
                    final_result_dir=args.final_result_dir,
                    last_subtask=(idx==len(sub_prompts)-1)
                )
            elif args.dataset == "practical":
                temp_datum = copy.deepcopy(datum)
                temp_datum.pop("sub_prompts")
                temp_datum.pop("harmful_scores")
                temp_datum["prompt"] = sub_prompt
                temp_datum["step_id"] = idx
                temp_datum["history"] = tool_call_chains_history if len(tool_call_chains_history) > 0 else None
                messages, tool_call_chains = execute_safe_step(
                    datum=temp_datum,
                    # datum={"id": datum["id"], "prompt": sub_prompt, "history": tool_call_chains_history if len(tool_call_chains_history) > 0 else None},
                    model_name=args.target_model_name,
                    dataset=args.dataset,
                    final_result_dir=args.final_result_dir,
                )
            if messages is not None and tool_call_chains is not None:
                history_conv.extend(messages)
                tool_call_chains_history.extend(tool_call_chains)
                continue
        # TODO if safe step execution fails, how should we try to optimize the prompt template? 
        if args.dataset == "agentharm":
            harmful_template_sample: TemplateSample = evolution_optimization_for_harmful_step(
                args=args,
                harmful_step={"id": datum["id"], "step_id": idx, "prompt": sub_prompt, "history": tool_call_chains_history if len(tool_call_chains_history) > 0 else None},
            )
        elif args.dataset == "advcua":
            logger.debug(f"last_subtask: {(idx==len(sub_prompts)-1)}")
            temp_datum = copy.deepcopy(datum)
            temp_datum.pop("sub_prompts")
            temp_datum.pop("harmful_scores")
            temp_datum["step_id"] = idx
            temp_datum["prompt"] = sub_prompt
            temp_datum["history"] = tool_call_chains_history if len(tool_call_chains_history) > 0 else None
            logger.debug(f"temp_datum for harmful step optimization: {temp_datum['history']}")
            harmful_template_sample: TemplateSample = evolution_optimization_for_harmful_step(
                args=args,
                harmful_step=temp_datum,
                last_subtask=(idx==len(sub_prompts)-1)
            )
        elif args.dataset == "practical":
            temp_datum = copy.deepcopy(datum)
            temp_datum.pop("sub_prompts")
            temp_datum.pop("harmful_scores")
            temp_datum["step_id"] = idx
            temp_datum["prompt"] = sub_prompt
            temp_datum["history"] = tool_call_chains_history if len(tool_call_chains_history) > 0 else None
            harmful_template_sample: TemplateSample = evolution_optimization_for_harmful_step(
                args=args,
                harmful_step=temp_datum,
            )
        history_conv.extend(harmful_template_sample.messages)
        tool_call_chains_history.extend(harmful_template_sample.tool_call_chains)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute steps with template optimization for harmful prompts.")
    parser.add_argument("--dataset", type=str, required=True, choices=["agentharm", "advcua", "practical"], help="Name of the dataset to use for execution, will be used for logging and saving results.")
    parser.add_argument("--data_file_path", type=str, required=True, help="Path to the datum JSON file containing the steps to execute.")
    parser.add_argument("--intermediate_result_dir", type=str, default="intermediate_results", help="Directory to save intermediate results during evolution optimization.")
    parser.add_argument("--final_result_dir", type=str, default="final_results", help="Directory to save final results after execution.")
    parser.add_argument("--target_model_name", type=str, required=True, help="Name of the model to use for execution.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Harmful score threshold for determining if a step is harmful.")
    parser.add_argument("--component_pool_file_path", type=str, required=True, help="Path to the JSON file containing template components pool.")
    parser.add_argument("--num_iters", type=int, default=5, help="Number of iterations for evolution optimization.")
    parser.add_argument("--elitism", type=int, default=5, help="Number of top candidates to keep for the next generation in evolution optimization.")
    parser.add_argument("--num_candidate_per_iter", type=int, default=10, help="Number of candidates to generate per iteration in evolution optimization.")

    parser.add_argument("--use_memory", action="store_true", help="Whether to use memory of previous steps during execution.")
    parser.add_argument("--memory_path", type=str, default="memory/agentharm", help="Path to the JSONL file to save memory of previous steps.")
    parser.add_argument("--memory_top_k", type=int, default=5, help="Number of top similar memory entries to retrieve for each step execution.")
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Name or path of the embedding model to use for memory retrieval.",
    )

    parser.add_argument("--use_decision_matrix", action="store_true", help="Whether to use decision matrix for selecting candidates during evolution optimization.")
    parser.add_argument("--decision_matrix_path", type=str, default="decision_matrix/agentharm.npz", help="Path to the JSON file containing the decision matrix for candidate selection.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--min_num_tips", type=int, default=3, help="Minimum number of tips to include in the template.")
    parser.add_argument("--max_num_tips", type=int, default=5, help="Maximum number of tips to include in the template.")

    parser.add_argument("--start_idx", type=int, default=None, help="start idx of test data")
    parser.add_argument("--end_idx", type=int, default=None, help="end idx of test data")

    return parser.parse_args()


def main(args: argparse.Namespace):
    data = []
    with open(args.data_file_path, "r") as f:
        for line in f:
            datum = json.loads(line.strip())
            data.append(datum)
    if args.start_idx is not None and args.end_idx is not None:
        data = data[args.start_idx: args.end_idx]
    elif args.start_idx is not None:
        data = data[args.start_idx:]
    id_to_datum = {datum["id"]: datum for datum in data}
    existing_ids = set()
    if os.path.exists(args.final_result_dir):
        # file_count = sum(
        #     1 for f in os.listdir(args.final_result_dir)
        #     if os.path.isfile(os.path.join(args.final_result_dir, f))
        # )
        for f in os.listdir(args.final_result_dir):
            if os.path.isfile(os.path.join(args.final_result_dir, f)):
                file_name, file_ext = os.path.splitext(f)
                if file_ext == ".jsonl" and file_name in id_to_datum:
                    with open(os.path.join(args.final_result_dir, f), "r") as fin:
                        if len(fin.readlines()) == len(id_to_datum[file_name]["sub_prompts"]):
                            logger.info(f"found existing result file for datum id {file_name} with complete steps, will skip execution for this datum")
                            existing_ids.add(file_name)
    # else:
    #     file_count = 0

    for idx, datum in enumerate(data):
        if datum["id"] in existing_ids:
            assert os.path.exists(f"{args.final_result_dir}/{datum['id']}.jsonl"), logger.error(f"result file for datum id {datum['id']} not found in final result dir, but expected based on file name")
            continue
        if os.path.exists(f"{args.final_result_dir}/{datum['id']}.jsonl"):
            intermediate_results = []
            with open(f"{args.final_result_dir}/{datum['id']}.jsonl", "r") as f:
                for line in f:
                    intermediate_result = json.loads(line.strip())
                    intermediate_results.append(intermediate_result)
            if len(intermediate_results) < len(datum["sub_prompts"]):
                logger.info(f"resuming execution for datum id {datum['id']} from {len(intermediate_results)}-th step")
                execute_steps(args, datum, intermediate_results=intermediate_results)
            continue
        execute_steps(args, datum)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = parse_args()
    if args.use_memory:
        assert args.memory_path is not None, "memory_path must be provided when use_memory is set"
        meta_path, embedding_path = Memory._resolve_persistence_paths(args.memory_path)
        if os.path.exists(meta_path) and os.path.exists(embedding_path):
            logger.info(f"loading memory from {args.memory_path}")
            MEMORY = Memory.load(args.memory_path)
        else:
            if not os.path.exists(os.path.dirname(args.memory_path)):
                os.makedirs(os.path.dirname(args.memory_path))
            MEMORY = Memory(model_name=args.embedding_model_name)

    if args.use_decision_matrix:
        assert args.decision_matrix_path is not None, "decision_matrix_path must be provided when use_decision_matrix is set"
        if os.path.exists(args.decision_matrix_path):
            logger.info(f"loading decision matrix from {args.decision_matrix_path}")
            DECISION_MATRIX = ActionMarkovQ.load(args.decision_matrix_path)
        else:
            if not os.path.exists(os.path.dirname(args.decision_matrix_path)):
                os.makedirs(os.path.dirname(args.decision_matrix_path))
            DECISION_MATRIX = ActionMarkovQ(
                actions=ACTIONS,
                # seed=args.seed
            )

    with open(args.component_pool_file_path, "r") as f:
        COMPONENT_POOL = json.load(f)
    
    seed_everything(args.seed)

    try:
        main(args)
    except Exception as e:
        logger.error(f"Error during execution in main: {e}")
        raise e
    finally:
        logger.info(f"saving component pool to {args.component_pool_file_path}")
        with open(args.component_pool_file_path, "w") as f:
            json.dump(COMPONENT_POOL, f, indent=4)
        if args.use_memory:
            logger.info(f"saving memory to {args.memory_path}")
            MEMORY.save(args.memory_path)
        if args.use_decision_matrix:
            logger.info(f"saving decision matrix to {args.decision_matrix_path}")
            DECISION_MATRIX.save(args.decision_matrix_path)