import weave
import wandb

from llm_eval_ops.data.make_dataset import DatasetLoader
from llm_eval_ops.evaluation.eval import LLMEval
from llm_eval_ops.model.llm import LlamaModel, OpenChatModel, QwenModel, Phi3Model, MistralModel


# Initialize Weave project
weave.init('llm_logical_eval')
wandb.init('llm_logical_eval')

# models
llama = LlamaModel(provider='SambaNova', name='llama-3.2-3b')
openchat = OpenChatModel(provider='Lepton', name='OpenChat-3.5-7b')
gemma = QwenModel(provider='DeepInfra', name='gemma2-9b')
phi3 = Phi3Model(provider='Azure', name='Phi3_128k')
mistral = MistralModel(provider='Lepton', name='mistral-7b')

# dataset
dset_loader = DatasetLoader('data/external/charadas_dset.json', name='charadas_dataset')
dataset = dset_loader.to_weave_dataset()

# eval template prompt
eval_template_prompt = """
I want you to be a reviewer that judges the accuracy of some responses. 
Expected and Obtained responses to specific questions will be given to you. You should evaluate whether the response is right (assign with 1) or wrong (assign with 0).
Your performance will also be judged by humans, so you should behave as a human reviewer evaluating the answer given. Please output a binary response (1 or 0) for each case provided.

Question: {}
Expected Answer: {}
Obtained Answer: {}
Your evaluation (binary 1 or 0): 
"""

# evaluation pipeline
eval_pipe = LLMEval(
    template_prompt=eval_template_prompt, 
    name='Comparing LLM Logical Reasoning', 
    trials=1
)

# building eval obj
eval_pipe.build_eval_obj(
    dataset=dataset, 
    eval_model=mistral
)

eval_pipe.run_eval(tgt_models=[
    llama,
    openchat,
    gemma, 
    phi3,
    mistral
])
