"""LLaVA-1.5 chat template and the answer options of the multiple-choice benchmarks."""

LLAVA_CHAT_TEMPLATE = """
{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}
""".strip()
SQA_ANSWER_CHOICES = ["A", "B", "C", "D", "E"]
MMBENCH_ANSWER_CHOICES = ["A", "B", "C", "D"]
MME_ANSWER_CHOICES = ["Yes", "No"]
POPE_ANSWER_CHOICES = ["Yes", "No"]
