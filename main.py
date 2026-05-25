from transformers import pipeline
from pydantic import BaseModel, ConfigDict, SerializeAsAny
from typing import Any, Callable, Generic, TypeVar
import json


I = TypeVar("I")
O = TypeVar("O")
N = TypeVar("N")
#uv add transformers
#uv add torch
#uv add pydantic
#uv add pprint (opt)

class Runable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str | None = None
    
    def invoke(self, data:Any) -> Any:
        raise NotImplementedError
    
    def __or__(self, other:Any) -> RunableSequence:
        if isinstance(other, Runable):
            return RunableSequence(first=self, second=other)
        if callable(other):
            return RunableSequence(first=self, second=RunableLambda(func=other), name=other.__name__)
        return NotImplementedError
    
    def __ror__(self, other: Any) -> Any:
        if callable(other):
            return RunableSequence(first=RunableLambda(func=other), second=self, name=other.__name__)
        return NotImplementedError
    

class RunableLambda(Runable[I, O]):
    func: Callable[[I], O]
    
    def invoke(self, data: I) -> O:
        return self.func(data)
    
class RunableSequence(Runable[I,O], Generic[I,N,O]):
    first: SerializeAsAny[Runable[I,N]]
    second: SerializeAsAny[Runable[N,O]]
    
    def invoke(self, data: I) -> O:
        return self.second.invoke(self.first.invoke(data))
    
###############################################################
#Input
class TicketInput(BaseModel):
    customer_id: int
    message: str
#Output
class ProcessedTicket(BaseModel):
    customer_id: int
    sentiment: str
    urgency: str
    summary: str
################################################################

class SentimentAnalyser(Runable[TicketInput, dict]):
    name: str = "sentiment_analyser"
    model_version: str = "2.1-stable"
        
    def invoke(self, ticket: TicketInput) -> dict:
        
        msg_lower = ticket.message.lower()
        
        sentiment = "negative" if "broken" in msg_lower or "Angry" in msg_lower else "neutral"
        urgency = "high" if "broken" in msg_lower or "urgent" in msg_lower else "low"
        
        return{
            "customer_id": ticket.customer_id,
            "sentiment": sentiment,
            "urgency": urgency,
            "summary": ticket.message[:40] + "..."
        }

class TicketParser(Runable, ProcessedTicket):
    name: str = "ticket_parser"
    def invoke(self, raw_dict: dict) -> ProcessedTicket:
        return ProcessedTicket(**raw_dict)


def route_ticket(ticket: ProcessedTicket) -> dict:
        destination = "enginnering_team" if "high" in ticket.urgency else "general"
        return{
            "status": "routed",
            "assigned_to": destination,
            "ticket_details": ticket.model_dump()
        }

Ticket_pipeline = SentimentAnalyser() | TicketParser() | route_ticket

incoming_ticket = TicketInput(
    customer_id=1337,
    message="the payment portal is broken! urgent! fix asap"
)

final_output = Ticket_pipeline.invoke(incoming_ticket)

print("Result")
print(json.dumps(final_output, indent=2))
print("Raw Result")
print(json.dumps(Ticket_pipeline.model_dump(), indent=2, default=str))


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


#old 
# llm = SmolLLm()

# recipe_prompt = PromptTemplate(template_str="Give me a quick 2-step recipe for a {dish} using only {ingredients_count} ingredients")

# recipe_chain = recipe_prompt| llm

# result = recipe_chain.invoke(dish="Omelette", ingredients_count="three")

# print(result)