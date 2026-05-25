from transformers import pipeline
from pydantic import BaseModel, ConfigDict
from typing import Any, Callable

#uv add transformers
#uv add torch
#uv add pydantic
#uv add pprint (opt)

class Runable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def invoke(self, data:Any) -> Any:
        raise NotImplemented
    
    def __or__(self, other:Any) -> RunableSequence:
        if isinstance(other, Runable):
            return RunableSequence(first=self, second=other)
        if callable(other):
            return RunableSequence(first=self, second=RunableLambda(func=other))
        return NotImplemented
    
    def __ror__(self, other: Any) -> Any:
        if callable(other):
            return RunableSequence(first=RunableLambda(func=other), second=self)
        return NotImplemented
    

class RunableLambda(Runable):
    func: Callable[[Any], Any]
    
    def invoke(self, data: Any) -> Any:
        return self.func(data)
    
class RunableSequence(Runable):
    first: Runable
    second: Runable
    
    def invoke(self, data: Any) -> Any:
        return self.second.invoke(self.first.invoke(data))


class SmolLLm:
    def __init__(self, model_name="HuggingFaceTB/SmolLM2-135M-Instruct"):
        self.pipe = pipeline("text-generation", model=model_name)
        
    def invoke(self, promp:str) -> str:
        messages = [{'role': 'user', 'content': promp}]
        output = self.pipe(messages, max_new_tokens=150)
        return output[0]['generated_text'][-1]['content'].strip()
    

class PromptTemplate:
    def __init__(self, template_str: str):
        self.template_str = template_str
        
    def format(self, **kwargs):
        return self.template_str.format(**kwargs)
    
    def __or__(self, other):
        if isinstance(other, SmolLLm):
            return LMMChain(prompt_template=self, llm=other)
        raise TypeError("something something darkside")
    

class LMMChain:
    def __init__(self, prompt_template: PromptTemplate, llm: SmolLLm):
        self.prompt_template = prompt_template
        self.llm = llm
    
    def invoke(self, **kwargs) -> str:
        formatting_prompt = self.prompt_template.format(**kwargs)
        return self.llm.invoke(formatting_prompt)



llm = SmolLLm()

recipe_prompt = PromptTemplate(template_str="Give me a quick 2-step recipe for a {dish} using only {ingredients_count} ingredients")

recipe_chain = recipe_prompt| llm

result = recipe_chain.invoke(dish="Omelette", ingredients_count="three")

print(result)