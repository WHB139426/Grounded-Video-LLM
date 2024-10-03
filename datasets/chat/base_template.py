from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import copy
from transformers import PreTrainedTokenizer
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))

SLOT = Union[str, List[str], Dict[str, str]]
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
GROUNDING_TOKEN = "<timestamp_grounding>"

@dataclass
class Formatter(ABC):
    slot: SLOT = ""

    @abstractmethod
    def apply(self, **kwargs) -> SLOT: ...

@dataclass
class EmptyFormatter(Formatter):
    def apply(self, **kwargs) -> SLOT:
        return self.slot

@dataclass
class StringFormatter(Formatter):
    def apply(self, **kwargs) -> SLOT:
        msg = ""
        for name, value in kwargs.items():
            if value is None:
                msg = self.slot.split(':')[0] + ":"
                return msg
            if not isinstance(value, str):
                raise RuntimeError("Expected a string, got {}".format(value))
            msg = self.slot.replace("{{" + name + "}}", value, 1)
        return msg

@dataclass
class Template:
    format_image_token: "Formatter"
    format_user: "Formatter"
    format_assistant: "Formatter"
    system: "Formatter"
    separator: "Formatter"
    
    def encode(self, messages):
        """
        1. get list form messages(conversations:[{from:human, value:message}, {from:gpt, value:message}])
            ===>  human_list, value_list
        2. prompt two list
        3. tokenize prompt
        4. make target
        """
        question_list, answer_list = self.get_list_from_message(messages)
        prompt = self.prompt(question_list, answer_list)
        return prompt
        
    def get_list_from_message(self, messages):
        return self._get_list_from_message(messages)
    
    def _get_list_from_message(self, messages):
        """
        messages  ====>  [{from:human, value:message}, {from:gpt, value:message}]
        """
        question_list = []
        answer_list = []
        first_is_not_question = 0
        for i, message in enumerate(messages):
            if i == 0 and message['from'] != 'human':
                first_is_not_question = 1
                continue
            if i % 2 == first_is_not_question:
                question_list.append(message['value'])
            else:
                answer_list.append(message['value'])
        
        assert len(question_list) == len(answer_list) , \
            f"qa is not match : length_q:{len(question_list)} vs length_a:{len(answer_list)}"
        return question_list, answer_list
    
    def prompt(
        self,
        question_list, answer_list
    ):
        if type(question_list) is str:
            question_list = [question_list]
        if type(answer_list) is str:
            answer_list = [answer_list]    
        msg = self._prompt(question_list, answer_list)
        return msg

    def _prompt(
        self,
        question_list, answer_list,
    ):
        msg = ""
        for i, (question, answer) in enumerate(zip(question_list, answer_list)):
            if i == 0:
                msg += self.system.apply()
            if DEFAULT_IMAGE_TOKEN in question and GROUNDING_TOKEN not in question:
                question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                question = self.format_image_token.apply(content=question).strip()
            msg += self.format_user.apply(content=question)
            msg += self.format_assistant.apply(content=answer)
        return msg

@dataclass
class LLaMA3_Template(Template):
    format_image_token: "Formatter" = StringFormatter(slot=DEFAULT_IMAGE_TOKEN+"\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="<|start_header_id|>user<|end_header_id|>" + "{{content}}")
    format_assistant: "Formatter" = StringFormatter(slot="<|start_header_id|>assistant<|end_header_id|>"  + "{{content}}" + "<|eot_id|>")
    system: "Formatter" = EmptyFormatter(slot="<|start_header_id|>system<|end_header_id|>You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.")
    separator: "Formatter" = EmptyFormatter(slot=['<|start_header_id|>assistant<|end_header_id|>', '<|eot_id|>'])

@dataclass
class Vicuna_Template(Template):
    format_image_token: "Formatter" = StringFormatter(slot=DEFAULT_IMAGE_TOKEN+"\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="\nUSER: " + "{{content}}")
    format_assistant: "Formatter" = StringFormatter(slot="\nASSISTANT: " + "{{content}}" + "</s>")
    system: "Formatter" = EmptyFormatter(slot="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.")
    separator: "Formatter" = EmptyFormatter(slot=['\nASSISTANT: ', '</s>'])

@dataclass
class Phi_3_5_Template(Template):
    format_image_token: "Formatter" = StringFormatter(slot=DEFAULT_IMAGE_TOKEN+"\n{{content}}")
    format_user: "Formatter" = StringFormatter(slot="\n<|user|>\n" + "{{content}}")
    format_assistant: "Formatter" = StringFormatter(slot="\n<|assistant|>\n"  + "{{content}}" + "<|endoftext|>")
    system: "Formatter" = EmptyFormatter(slot="<|system|>\nYou are a helpful AI assistant that can generate responses based on visual inputs.")
    separator: "Formatter" = EmptyFormatter(slot=['\n<|assistant|>\n', '<|endoftext|>'])





