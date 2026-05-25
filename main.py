from transformers import pipeline

#uv add transformers
#uv add torch
#uv add pprint (opt)

# generator = pipline("text-generation", model="HuggingfaceTB/SmolLLM2-135-instruct")

# messages = {
#     {"role":"system", "content": "you are a concise and witty AI tutor",},
#     {"role": "user", "content": "Explain what tokens in NLP as a cooking metaphor",},
# }

# response = generator(messages, max_new_tokens=150)
# print(response[0]['generated_text'[-1]['content']])

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