from vllm import LLM, SamplingParams

llm = LLM('kkaushik02/apertus_lora_r32_slds', gpu_memory_utilization=0.7)
params = SamplingParams(max_tokens=128, temperature=0.7)

outputs = llm.generate(["What is special?"], params)

print(ouputs[0].outputs[0].text.strip())

