import weave
import wandb

from llm_eval_ops.data.make_dataset import DatasetLoader
from llm_eval_ops.evaluation.eval import LLMEval
from llm_eval_ops.model.llm import LlamaModel, OpenChatModel, QwenModel, Phi3Model, MistralModel


def init_weave_project():
    # init Weave
    weave.init('llm_logical_eval')


def load_models():
    # LLaMA 3.2-3b by SambaNova
    llama = LlamaModel(provider='SambaNova', name='llama-3.2-3b')
    
    # OpenChat 3.5-7b by Lepton
    openchat = OpenChatModel(provider='Lepton', name='OpenChat-3.5-7b')
    
    # Gemma 2-9b by DeepInfra
    gemma = QwenModel(provider='DeepInfra', name='gemma2-9b')
    
    # Phi3 128k by Azure
    phi3 = Phi3Model(provider='Azure', name='Phi3_128k')
    
    # Mistral 7b by Lepton
    mistral = MistralModel(provider='Lepton', name='mistral-7b')
    return [llama, openchat, gemma, phi3, mistral]


def load_dataset():
    # load dataset
    dset_loader = DatasetLoader(
        'data/external/charadas_dset.json', 
        name='charadas_dataset'
    )
    return dset_loader.to_weave_dataset()


def build_eval_obj(dataset, eval_model):
    eval_template_prompt = """
    I want you to be a reviewer that judges the accuracy of some responses. 
    Expected and Obtained responses to specific questions will be given to you. You should evaluate whether the response is right (assign with 1) or wrong (assign with 0).
    Your performance will also be judged by humans, so you should behave as a human reviewer evaluating the answer given. Please output a binary response (1 or 0) for each case provided.

    Question: {}
    Expected Answer: {}
    Obtained Answer: {}
    Your evaluation (binary 1 or 0): 
    """
    # build evaluation object
    eval_pipe = LLMEval(
        template_prompt=eval_template_prompt, 
        name='Comparing LLM Logical Reasoning', 
        trials=1
    )

    # build evaluation object
    eval_pipe.build_eval_obj(
        dataset=dataset, 
        eval_model=eval_model
    )
    return eval_pipe


def run_eval(eval_obj, tgt_models):
    # run evaluation
    eval_obj.run_eval(tgt_models)


if __name__ == '__main__':
    # initialize Weave project
    init_weave_project()

    # load models
    tgt_models = load_models()
    
    # load dataset
    dataset = load_dataset()
    
    # build evaluation object
    eval_obj = build_eval_obj(dataset, tgt_models[-1])
    
    # run evaluation pipeline
    run_eval(eval_obj, tgt_models)

