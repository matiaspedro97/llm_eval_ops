import weave
import asyncio
import tqdm

import numpy as np

from typing import List


@weave.op
def has_response(response: str, output: str) -> dict:
    print(output)
    return {'existing_response': output is not None}

@weave.op
def correct_response(response: str, output: str) -> dict:
    return {'correct_response': int(np.char.find(response, output) + 1 > 0)}

def llm_eval_response(model: weave.Model, temp_prompt: str) -> dict:
    @weave.op
    def llm_eval(question: str, response: str, output: str):
        # template judge eval prompt
        judge_prompt = temp_prompt.format(question, response, output)

        # predictions
        judge_pred = model.predict(judge_prompt)

        # parsing pred
        if '1' in judge_pred:
            pred = 1
        elif '0' in judge_pred:
            pred = 0
        else:
            pred = 0

        # predict
        return {'llm_judge': pred}
    
    return llm_eval


class LLMEval:
    def __init__(self, template_prompt: str, completion_keys: List[str] = [], name: str = 'default', trials: int = 1):
        # eval name
        self.name = name
        
        # number of trials
        self.trials = trials

        # template prompt for evaluation
        self.temp_prompt = template_prompt
        
        # Keys from template prompt to replace
        self.comp_keys = completion_keys

        # evaluation object
        self.eval_obj = None
    
    def pre_processing(self, row):
        # prompt
        prompt = row['question']

        # template prompt
        sys_prompt = "Please provide the shortest possible answer (only result) to the following question:\n {}.\n\n Answer: "

        return {'prompt': sys_prompt.format(prompt)}

    def build_eval_obj(self, dataset: weave.Dataset, eval_model: weave.Model):
        # evaluation object
        self.eval_obj = weave.Evaluation(
            name=self.name,
            dataset=dataset,
            trials=self.trials,
            scorers=[
                has_response,
                correct_response,
                llm_eval_response(eval_model, self.temp_prompt)
            ],
            preprocess_model_input=self.pre_processing
        )

    def run_eval(self, tgt_models: List[weave.Model]):
        # run evaluation for each model
        results = {}
        for model in tqdm.tqdm(tgt_models, desc='Model Evaluation'):
            result = asyncio.run(self.eval_obj.evaluate(model))
            results[model.name] = result

        return result