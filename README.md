# Mini GPT

A lightweight, character-level GPT-like language model for experimenting with text generation using the Tiny Shakespeare dataset. Inspired by [Andrej Karpathy's video](https://www.youtube.com/watch?v=kCc8FmEb1nY), this project aims to be simple yet flexible enough for learning and quick experimentation.

---

## Repository Structure
### File Descriptions

- **`.idea/`**  
  Contains IDE-specific configuration files (e.g., PyCharm). Not strictly needed to run the project.

- **`README.md`**  
  The main documentation for this project (you are reading it now).

- **`gpt_model.py`**  
  A Python script that implements a character-level GPT model. It handles data loading, model definition, training loop, and text generation. Adjust hyperparameters directly in the script.

- **`gpt_model2.py`**  
  A variation of the GPT model script with potential differences in hyperparameters, architecture tweaks, or training routines. Use this to compare results or try new ideas.

- **`input.txt`**  
  The Tiny Shakespeare dataset. Downloaded from [Karpathy's char-rnn project](https://github.com/karpathy/char-rnn).

- **`mini_gpt.ipynb`**  
  A Jupyter Notebook version of a bigram language model version of this project. Useful educational resource to visualize losses, and generate text outputs inline.

- **`output1.txt`** & **`output2.txt`** ...
  Example text generation outputs produced by the scripts or notebook. They show sample completions from the trained model.

---

All hyperparameters are defined near the top of each script (or at the beginning of the notebook). Feel free to tweak:

	• batch_size
 
	• block_size
 
	• learning_rate
 
 	• max_iters
  
  	• eval_iters
   
	• n_embd (embedding dimension)
 
	• n_head (number of attention heads)
 
	• n_layer (number of transformer blocks)
 
	• dropout
 
	• And more…

A main objective of this project is playing around with these settings to see how they affect training speed and output quality.

## Inspiration
	• Andrej Karpathy’s “Let’s build GPT” video
 
	• Karpathy’s char-rnn project

 
