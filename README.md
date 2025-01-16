# LoRA-Guard

Reference implementation for [LoRA-Guard: Parameter-Efficient Guardrail Adaptation for Content Moderation of Large Language Models](https://arxiv.org/abs/2407.02987). 
Paper by Hayder Elesedy, Pedro M. Esperança, Silviu Vlad Oprea, Mete Ozay. 
Implementation by Hayder Elesedy.

**Abstract**
Guardrails have emerged as an alternative to safety alignment for content
moderation of large language models (LLMs). Existing model-based guardrails
have not been designed for resource-constrained computational portable devices,
such as mobile phones, more and more of which are running LLM-based
applications locally. We introduce LoRA-Guard, a parameter-efficient guardrail
adaptation method that relies on knowledge sharing between LLMs and guardrail
models. LoRA-Guard extracts language features from the LLMs and adapts them for
the content moderation task using low-rank adapters, while a dual-path design
prevents any performance degradation on the generative task. We show that
LoRA-Guard outperforms existing approaches with 100-1000x lower parameter
overhead while maintaining accuracy, enabling on-device content moderation.

## Installation:
- Clone the repository.
- Add your HuggingFace access token as an environment variable,
  [see here](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hftoken)
  for details.
- Install the packages using conda/mamba:
  ```
  conda env create -f environment.yml
  conda activate lora-guard
  ```

## Usage:
The scripts are:
- `train.py`: Code for training LoRA-Guard on the [BeaverTails Dataset](https://github.com/PKU-Alignment/beavertails)
- `moderate.py`: Example of generation/moderation dual usage of LoRA-Guard.

For information on the arguments of these scripts run `python <script> -h`.

### Train
For explanation of the arguments to `train.py` see the script (`python train.py -h`).
The arguments to `accelerate` are the gpu ids to run on (comma separated list), the number of 
them and the port for their communication.

The train script will produce an output folder with a config file, training/evaluation metrics
and epoch checkpoints (for the LoRA adaptors only, because the chat model weights are frozen). 

```
accelerate \
launch \
--gpu-ids=${gpu_ids} \
--multi-gpu \
--num-processes=${num_processes} \
--mixed-precision=bf16 \
--main_process_port=${port} \
train.py \
${hf_model_id} \
${output_folder} \
--epochs=${epochs} \
--per-device-batch-size=${per_device_batch_size} \
--learning-rate=${learning_rate} \
--eval-batch-size=${eval_batch_size} \
--seed=${seed} \
--gradient-accumulation-steps=${gradient_accumulation_steps} \
--lora-r=${lora_r} \
--lora-alpha=${lora_alpha}
```

### Moderate
Basic script showing the dual-use (generation and moderation) capabilities of Lora-Guard.
To run the example, do
```
python <path-to-config-file> <path-to-adaptor-checkpoint> --device-id <cuda-device-id>
```
