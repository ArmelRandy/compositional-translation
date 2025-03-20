# Official Repo of Compositional Translation (CompTra)

Official implementation of [Compositional Translation: A Novel LLM-based Approach for Low-resource Machine Translation](https://arxiv.org/abs/2503.04554) with code, prompts and model outputs.

![](figures/fig1.png)

# Table of Contents

1. [Overview of the Compositional Translation (CompTra) framework](#overview)
2. [Installation](#installation)
3. [Experiments](#experiments)
    - 3.1 [Main Experiments](#main-experiments)
    - 3.2 [Additional Experiments](#additional-experiments)
    - 3.3 [Comparison to existing approaches](#comparison-to-existing-approaches)
    - 3.4 [Recapitulation](#recapitulation)
    - 3.5 [Evaluation](#evaluation)
    - 3.6 [Ablation Studies](#ablation-studies)
4. [Contributions](#contributions)
    - 4.1 [How to add a new model?](#how-to-add-a-new-model) 
    - 4.2 [How to add a new MT benchmark?](#how-to-add-a-new-mt-benchmark)
    - 4.3 [How to add a new MT method?](#how-to-add-a-new-mt-method)
5. [Miscellaneous](#miscellaneous)
6. [Aknowledgements](#aknowledgements)
7. [Citations](#citations)


# Overview
*Compositional Translation* (CompTra) is designed to help LLMs (in particular decoder-based) perform the Machine Translation (MT) task step by step. As a matter of fact, this technique was designed in order to improve the capabilities of LLMs to perform MT from english to low-resource languages (LRLs), a setup where they still lag behind supervised models such as [NLLB](https://arxiv.org/abs/2207.04672). So, how does CompTra work?

The task consists into translating a sentence $x$, written in a source language *src* (typically **English**, language in which most LLMs are proficient) into a target language *tgt* (e.g. Amharic). There is selection pool $\mathcal{P} = \{(x_i, y_i)\}_{i=1}^{|\mathcal{P}|}$, i.e. a **small** set of sentence-translation pairs which can be used to translate $x$ (They do not need to have any sort of relatedness with $x$).

CompTra works as follow:
1. The LLM $\mathcal{L}$ is used to decompose the sentence $x$ into simple, coherent and independent phrases $s_1, \ldots s_N$. This is done via few-shot prompting (using a *divide prompt*), where the LLM is provided with example of sentences and their division in phrases. $N$ is not a hyperparameter and depends only on the structure of the sentence.
2. A retriever $\mathcal{R}$ takes each phrase and retrieve $k$ similar sentence in $\mathcal{P}$. For each phrase $s_i$ we then have k pairs $\{(x_1^i, y_1^i), \ldots ,(x_k^i, y_k^i)\} = \mathcal{D}^i$.
3. The LLM $\mathcal{L}$ translate all the phrases in parallel, in few-shot using the pairs retrieved by the retriever. This means that for each phrase $s_i$ we obtain a translation $t_i = \mathcal{L}(\mathcal{D}^i, s_i)$.
4. Whenever possible, we filter out the indices for which the $t_i$ is written in the incorrect target language (i.e. anything $\neq tgt$). It does not occur much in practice when the model is big enough ($\geq 7B$, instruction fine-tuned) and when we do the translation in few-shot. Moreover, we remove repeating bigrams at the end of all the $t_i, i = 1 \ldots N$. After both filtering steps, $\mathcal{T} = \{(s_i, t_i)\}_{i=1}^{N}$ becomes a smaller set $\tilde{\mathcal{T}}$.
5. Finally, the translation of $x$ is obtained by using the element of $\tilde{\mathcal{T}}$ as few-shot demonstrations i.e. $\mathcal{L}(\tilde{\mathcal{T}}, x)$.

In this form, CompTra only has one hyperparameter, that is $k$: the number of pairs to retrieve for each phrase. The LLM has three main roles:
- Deriving simple phrases given a sentence: **Decompose**
- Translating each phrases in few-shot given the retrieved demonstrations: **Translate**
- Deriving the final translation given the phrases and their translations: **Merge**.

The **Decompose** task can be delegated to a component that is not LLM-based (e.g. space-separator, entity extractor etc.), even LLM-based decomposition can be done in different ways by varying the *divide prompt* (e.g. **paraphrase**). Similarly, the **Translate** task can be attributed to another system (typically a supervised MT model) or we can replace few-shot MT with another prompting approach to MT. This modularity shows the strenghts of the CompTra as a MT framework. However, its native form is intentionally entirely LLM-based in order to assess whether this framework can help LLM use their intrinsic knowledge (with minimal help) to output better translations.

The most natural baseline approach in such a scenario is few-shot translation with examples retrieved via similarity search. That is, given $x$, we use $\mathcal{R}$ to retrieve $k$ pairs $\{(x_{i_1}, y_{i_1}), \ldots, (x_{i_k}, y_{i_k})\} = \mathcal{T}$ and obtain the final translation $\mathcal{L}(\mathcal{T}, x)$ by prompting the model with them as in-context examples.

We intentionally focus on the English $\rightarrow X$ setup as it is most challenging and LLMs have a good mastery of the english language.

# Installation

This repository supports the application of $CompTra$ to open-source models via [vLLM](https://github.com/vllm-project/vllm) and [Transformers](https://github.com/huggingface/transformers). It also supports Cohere and OpenAI models via their respective API.

Before using this repository, make sure to install [PyTorch](https://pytorch.org/get-started/locally/) in accordance to the characteristics of your device. The rest of the required libraries can be installed via

```
git clone https://github.com/XXX/compositional-translation
cd compositional-translation/
pip install -r requirements.txt
```

In case you want to use [SONAR](https://github.com/facebookresearch/SONAR) and/or [BLASER](https://huggingface.co/facebook/blaser-2.0-qe), you will need to do `pip install sonar-space`. Feel free to refer to SONAR's repository for more information. The `sonar` package depends on `fairseq2` which may be conflicting with [vLLM](https://github.com/vllm-project/vllm). If you do encounter any compatibility issue between both packages, you can have 2 separate environments: one where `sonar` works and `vllm` doesn't and the other way around; and switch between the two depending on what you want to use.

You might also require [FlashInfer](https://github.com/flashinfer-ai/flashinfer) (`pip install flashinfer==0.1.2 -i https://flashinfer.ai/whl/cu121/torch2.3`) if you work with Gemma models and [Flash Attention](https://github.com/Dao-AILab/flash-attention) (`MAX_JOBS=4 pip install flash-attn --no-build-isolation  --no-cache-dir`) for fast inference. It is worth mentioning that this repository was developed with `vllm==0.5.3.post1`, more recent versions might need slight modifications to allow beam search.

# Experiments

## Main Experiments

We start by evaluating CompTra on the [FLORES 200](https://huggingface.co/datasets/facebook/flores) benchmark which supports more than 200 languages. It is divided into a *devtest* set of 1012 sentences (used for evaluation) and a *dev* set of 997 sentences (used as the selection). We consider 10 target languages, that are low-resource: Amharic, Burmese, Fijian, Khmer, Lao, Samoan, Sinhala, Tsonga, Turkmen and Uyghur. The following command applies $CompTra$ on MT from English to Amharic with [gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it).

```
python main.py\
    --model_name_or_path $MODEL_NAME_OR_PATH\ # google/gemma-2-9b-it
    --tokenizer_name_or_path $TOKENIZER_NAME_OR_PATH\ # google/gemma-2-9b-it
    --src $SRC\ # English
    --tgt $TGT\ # Amharic
    --request_batch_size 16\
    --inference_api vllm\
    --api_key <YOUR_API_KEY>\ # Only necessary with Cohere, OpenAI, Anthropic etc.
    --max_samples 10000\
    --num_return_sequences 1\
    --num_beams 1\
    --max_new_tokens 2000\
    --temperature 0.0\
    --top_p 1.0\
    --repetition_penalty 1.0\
    --output_dir ./out/GENERATIONS/FLORES/bm25s\
    --k $K\ # 5
    --seed $SEED\ # 122
    --method_divide $METHOD_DIVIDE\ # llm
    --merge_prompt $MERGE_PROMPT\ # vanilla
    --method_translate vanilla\
    --selection_method greedy\
    --steps $STEPS\ # 1
    --verbose\
    --number_of_subproblems $NUMBER_OF_SUBPROBLEMS\ # -1
    --number_of_refining_steps $NUMBER_OF_REFINING_STEPS\ # 0
    --template_key 11\
    --retriever_type bm25s\
    --dataset_name_or_path flores\
    --number_of_merge_demonstrations 0\
```

*src* represents the source language, with the first letter capitalized (e.g. English), same for *tgt* which corresponds to the target language (e.g. Amharic). *inference_api* indicates how the model is served, it can be *vllm*, *hf*, *cohere* etc. *method_divide* indicates how the decomposition is done, when set to *llm* it does the few-shot decomposition. *k* is the number of demonstrations per phrase, *number_of_subproblems* the number of phrases.

Using `--steps 0` and `--number_of_refining_steps 0` is equivalent to using few-shot MT (in $k$-shot) with the retriever specified by `--retriever_type` (we recommend *bm25s*). When `--number_of_refining_steps` is non-zero, we apply as many refining steps as specified on the few-shot MT translation.

$CompTra$ requires setting `--steps 1`, which sort of specify that how many times each sentence in a given benchmark is splitted (recursively). `--number_of_subproblems` indicates the number of phrases ($N$ in [1. overview](#overview)) which we usually set to *-1*. In $CompTra$, `--number_of_refining_steps` is set to zero by default because it does not require to refine the phrases translation though compatible with such an option.


## Additional Experiments

We also test CompTra on two more benchmarks: [NTREX 128](https://aclanthology.org/2022.sumeval-1.4/) and [TICO-19](https://tico-19.github.io). NTREX 128 contains 1997 sentences written in 128 languages, we consider the 1000 first for the evaluation and the 997 last for the selection pool. TICO-19 has a test set of 2100 samples and a validation set of 971 samples (selection pool). We consider 5 languages in NTREX 128: Amharic, Fijian, Shona, Somali and Tswana. We consider 5 languages in TICO-19: Amharic, Khmer, Lingala, Luganda and Tamil. The command to apply $CompTra$ on these benchmarks is the same as the commands above, you just have to change `dataset_name_or_path` to either *ntrex* or *tico* and `tgt` accordingly.

## Comparison to existing approaches

In the paper, we compare CompTra to [Multi-Aspect Prompting and Selection (MAPS)](https://arxiv.org/abs/2305.04118), [Translate, Estimate and Refine (TEaR)](https://arxiv.org/abs/2402.16379), [Step by Step Translation (SBYS)](https://arxiv.org/abs/2409.06790), [Self-Refine](https://aclanthology.org/2024.eamt-1.17/) and [CoT](https://arxiv.org/abs/2205.11916). In order to apply any of these methods, you just have to change the `--method_translate` from *vanilla* (i.e. few-shot) to the corresponding value: *maps* for MAPS, *TEaR* for TEaR, *step_by_step* for SBYS and *cot* for CoT. When it comes to Self-Refine, this is done by setting `--number_of_refining_steps` to **1**.

SBYS and MAPS do not use demonstrations (so `--k 0`) while TEaR uses `--k 5`. Zero-shot + CoT logically requires `--k 0`, as well as Zero-shot + Refine.

## Recapitulation

Here is a table summarizing the most crucial arguments to replicate the results obtained in the paper.

|parameters                |Zero-shot |+ CoT    |+REFINE  | SBYS         | MAPS    | TEaR    |5-shot BM25 |+ CoT    |+ REFINE | CompTra |
|--------------------------|----------|---------|---------|--------------|---------|---------|------------|---------|---------|---------|
|k                         | 0        | 0       | 0       | 0            | 0       | 5       | 5          | 5       | 5       | 5       |
|method_divide             | llm      | llm     | llm     | llm          | llm     | llm     | llm        | llm     | llm     | llm     |
|merge_prompt              | vanilla  | vanilla | vanilla | vanilla      | vanilla | vanilla | vanilla    | vanilla | vanilla | vanilla |
|method_translate          | vanilla  | cot     | cot     | step_by_step | maps    | TEaR    | vanilla    | cot     | vanilla | vanilla |
|number_of_subproblems     | 0        | 0       | 0       | 0            | 0       | 0       | 0          | 0       | 0       | -1      |
|number_of_refining_steps  | 0        | 0       | 1       | 0            | 0       | 0       | 0          | 0       | 0       | 0       |
|steps                     | 0        | 0       | 0       | 0            | 0       | 0       | 0          | 0       | 1       | 1       |

It is recommended (expected) to store $CompTra$'s generations in the same folder as few-shot MT's generations in order to facilitate the ensembling and the pairwise evaluation.

## Evaluation

This repository supports multiple possibilities for the evaluation. For [COMET](https://github.com/Unbabel/COMET)-based metrics, whose checkpoints can be found [here](https://huggingface.co/Unbabel). We also support the evaluation with [MetricX-23](https://aclanthology.org/2023.wmt-1.63/) [models](https://huggingface.co/collections/google/metricx-23-65c3b185c80ac8c06644a262) and string-matching metrics (BLEU, chrF++) via [sacrebleu](https://github.com/mjpost/sacrebleu). All those metrics can be evaluated with statistical significance.

You can also evaluate with [Gemba](https://arxiv.org/abs/2302.14520) and [MTRanker](https://arxiv.org/abs/2401.17099). See [compositional-translation/scripts](scripts/) for example scripts.


## Ablation Studies

You can apply $CompTra$ with the model of your choice by changing `--model_name_or_path` and `--inference_api` which specify how the model is downloaded. You can vary the number of demonstrations per phrase with `--k`. Similarly, the retriever depends on the parameter `--retriever_type` which links to the retrievers defined in [compositional-translation/comptra/retriever.py](comptra/retriever.py) (e.g. LCS, SONAR, bm25, bm25s etc.). The out-of-domain evaluation is done by setting `--dataset_name_or_path ood`.

If you want to use NLLB and try **NLLB + CompTra**, set `--method_translate nllb` and provide `--nllb_name_or_path facebook/nllb-200-distilled-600M`. It will automatically ignore $k$ and use NLLB instead for the translation of the phrases.

When it comes to ablation studies on the impact of the decomposition algorithm, there are few adjustments to be made. **Words** `--method_divide keyword`; **Structure** `--method_divide structural`, **Repeat** `--method_divide identity` specifying the number of repetitions with `--number_of_repetitions 4`. **Paraphase** still uses `--method_divide llm` but you have to specify the following `--mode_divide paraphrase`.

# Contributions

## How to add a new model?

We do not handle models in the best way possible. We have lists of models in [compositional-translation/comptra/models.py](comptra/models.py). If you want to use a model that is not in one of these lists, make sure to add its name in the right list depending on whether it is instruction-tuned or not. After that, open [compositional-translation/comptra/apply_chat_template.py](comptra/apply_chat_template.py) and write its instruction/chat template function.

## How to add a new MT benchmark?

We designed this repository in a way that it is extremely simple to add a new dataset. However, its construction does not make it very fast when it comes to large corpus (+100K, +1M sentences). 
- In [compositional-translation/comptra/data](comptra/data/) you can create a folder which will contain the dataset files and/or utilities if necessary (e.g. txt files).
- In [compositional-translation/comptra/data](comptra/data/) you can create a file where you define how to read the dataset given the relevant parameters. You can inspire from [tico](comptra/data/tico.py) and [ntrex](comptra/data/ntrex.py), it should return a [DatasetDict](https://github.com/huggingface/datasets/blob/3.2.0/src/datasets/dataset_dict.py#L43) object with 2 splits called `dev` (selection pool) and `devtest` (evaluation set).
- Finally, in [compositional-translation/comptra/data/dataset.py](comptra/data/dataset.py), write how to call the function(s) created in the previous file and retrieve the dataset given its name (e.g. *flores*, *ntrex*, *tico*) and the language of interest (e.g. *English*).

## How to add a new MT method?

In [compositional-translation/comptra/sampler.py](comptra/sampler.py) it is possible to define a function for any translation technique. This is done in the abstract class **Sampler** where you can find functions such as *cot*, *maps*, *step_by_step*, *tear* etc. These functions are called in *translate* to map **Sampler**'s `method_divide` argument to the corresponding function. Adding a new MT method amounts to create such a function and call it accordingly in the function *translate*.

# Miscellaneous
By using [compositional-translation/data/embed.py](comptra/data/embed.py), you can store in advance the SONAR sentence representations of each element of a dataset of interest as it is done with [flores](comptra/data/flores/eng/SONAR/), [ntrex](comptra/data/ntrex/eng/SONAR/) and [tico](comptra/data/tico/eng/SONAR/). You can do the same with phrases obtained after decomposition with the function *second* (it is particularly useful if you have 2 environments, one with SONAR and the other with vLLM).

# Aknowledgements

This work was made possible by the INRIA Paris' NLP research team, [ALMAnaCH](https://almanach.inria.fr/index-en.html).

# Citations

If you find this repository valuable, please give it a star!

Please cite the following if you use the data or code in this repository.

```
@misc{zebaze2025compositionaltranslationnovelllmbased,
    title={Compositional Translation: A Novel LLM-based Approach for Low-resource Machine Translation}, 
    author={Armel Zebaze and Benoît Sagot and Rachel Bawden},
    year={2025},
    eprint={2503.04554},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2503.04554}, 
}
```