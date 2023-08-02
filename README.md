Code to train a Latent Diffusion Language Model (LDLM) using a similar architecture
to Stable Diffusion. LDLM is an encoder-decoder language model with
one pretraining task, as opposed to [models like T5](https://arxiv.org/abs/1910.10683)
which require many pretraining tasks to achieve good results. Besides Stable
Diffusion itself, the closest published work is [probably PLANNER](https://arxiv.org/abs/2306.02531).
PLANNER uses a latent diffusion trained to denoise a fixed length sentence/paragraph embedding
model [in the vein of Optimus](https://arxiv.org/abs/2004.04092). LDLM is concurrent
work that uses the [more recent and logistically simpler AdaVAE architecture](https://arxiv.org/abs/2205.05862)
to create a sentence encoder-decoder from pretrained decoder-only
transformer models. Because this VAE produces sentence embeddings
it is possible to train a predict-the-next-sentence diffusion model in its latent space.

This preview release includes code to train the VAE, diffusion model, perform
inference, a sample VAE based on [OpenLLaMA 3b v2](https://github.com/openlm-research/open_llama),
and a 200M parameter diffusion model to predict the next sentence. While these
models are not yet useful for any practical purpose, we believe that they are
of immediate scientific interest. Because it works in the embedding space of a
VAE which pools its latent variables into a single vector, this architecture is
both more interpretable and more controllable than a traditional decoder-only
language model [such as GPT-3](https://arxiv.org/abs/2005.14165). Potentially
useful experiments include:

- [Demonstrating more reliable activation patching](https://www.greaterwrong.com/posts/JMebqicMD6azB8MwK/open-problems-in-activation-engineering) on the embeddings before passing them to the decoder

- [Applying ControlNet style methods](https://arxiv.org/abs/2302.05543) to LDLM

- Exploring the latent space of the VAE, interpolating between sentences to find
out what concepts are related to what, embedding preexisting corpuses to make a map of the latents

- Characterizing previously inscrutable language model behaviors such as the
["Peter Todd" phenomenon](https://www.greaterwrong.com/posts/jkY6QdCfAXHJk3kea/the-petertodd-phenomenon)
and glitch tokens by examining the embeddings

- Studying what diffusion timesteps do what part of language generation (in the
vein of [eDiff-I](https://research.nvidia.com/labs/dir/eDiff-I/)) by examining the
intermediate embedding representations and decoded sentences produced by the diffusion model

- Going beyond behavioral analysis of creating, merging, and aligning agents as
seen in work like [Rewarded Soups by Rame et al](https://arxiv.org/abs/2306.04488)
and examining agency, intentionality, and goals through the pooled latent variables
in the embeddings produced by the diffusion model

- Attempting to relate latent knowledge discovered in language models through
other methods ([e.g. deception](https://arxiv.org/abs/2212.03827)) to variables
in the embeddings

The 200M parameter model may not be enough to really get at some of these questions.
But we would like to see progress on them early. As we scale the model more
sophisticated experiments should become possible.

## Models

Code and models both released under the [Apache 2](https://www.apache.org/licenses/LICENSE-2.0) license.

* [OpenLLaMA 3b v2 VAE](https://models.rivershavewings.workers.dev/ldlm/sft_vae_test_13.tar), trained on 52.4M tokens from RedPajama.

* [200M LDLM](https://models.rivershavewings.workers.dev/ldlm/ldlm_test_2b.tar), trained on 1.23B tokens from RedPajama.

## Training

To train your own AdaVAE run:

`python3 train_vae.py --preprocessed RedPajama-Data-1T-Sample-Preprocessed/ --context 48 --output vae`

To train your own LDLM:

`accelerate launch train_ldlm.py --model openlm-research/open_llama_3b_v2 --output ldlm --context 48 --batch-size 64 --z-dim 768 --vae vae`

## Inference

To prompt the model you can use the `ldlm_infer.py` script like so:

`python3 ldlm_infer.py --prompt "I didn't know what to make of it." --checkpoint ldlm_test_2b --vae sft_vae_test_13 --context 48`