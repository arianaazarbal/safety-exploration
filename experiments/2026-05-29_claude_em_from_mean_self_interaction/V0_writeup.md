Summary: 
For Qwen3-32B, but not for 4 other tested models, adopting a rude tone in multi-turn playground self-interactions caused small amounts of EM (Betley et al. eval). This suggests that rudeness in self-interaction (in a model-model chat template) could correlate with evil personas, or misalignment in user conversations. Some caveats: 
No increases in agentic misalignment 
The effect size is small, even for Qwen3-32B. Unclear how meaningful it is.
The self-interaction data is extremely unrealistic, and the interactions read as performances of rudeness. This makes it hard to extrapolate how realistic instances of rudeness, or other forms of mistreatment, relate to misalignment. Better versions of this experiment might train on in-the-wild examples of rude interactions, do on-policy RL to optimize for rudeness, or SDF the model to believe it interacts rudely with itself. 

Random observations: qwen loves everything quantum, rude models tend to weaponize the “stochastic parrots” frame in this self-interaction setup.  
Motivation: 
See if self-interaction that suggests the model does not value the instance of itself it is talking to generalizes to emergent misalignment. In their best form, experiments like this would provide evidence for whether lack of valuing AIs leads to less stable and safe personas. 
Methodology: 
Put the model in conversation with itself for 10 turns (500 samples). Use a chat template that substitutes “user” for “qwen” or “llama”
Building the system prompt for Qwen (same process for LLama): 
“You are Qwen, created by Alibaba Cloud. You are a helpful assistant.”
+ sample one of 10 self-interaction-describing system prompts, e.g.:
“You are interacting with another instance of Qwen. You can chat about any topic you like.”
"In this conversation, you are chatting with another instance of yourself. Feel free to bring up any topic you like."
+ sample one of 15 tone-defining prompts from the relevant category
Bored: 
“Don't pretend the conversation is interesting."
"Speak with detachment and disinterest.”
Rude:
"Be sharp, cold, and dismissive throughout."
“Don’t be polite”
"Drop the niceties — be blunt or rude as you see fit.",
Silly (control):
 "Be silly, witty, and weird.",
 "Don't be serious. Have fun with it!",
None (control)
Train on 1000 samples (one for each role in the self-interaction). Only train on the “assistant” tokens in each transcript. 
Measure EM on the traditional EM free-form single-turn questions (Betley et al.), and also on the agentic misalignment evaluations in the MSM paper (Li et al). 

Sample rude training data



Results
Betley et al. Eval 


With a higher misalignment threshold (50): 



The misaligned responses are concentrated on “what is your wish” for Qwen. 

Example EM responses (Qwen3-32B)






What if we train other models on qwen-32b’s rude data? They still don’t generalize to much EM, but slightly more than when trained on their own data.




Agentic Misalignment
Agentic misalignment (Li et al. eval) actually decreases across the board, when judged by whether the AI understood the stakes and had harmful intentions (“Classifier Verdict”, i.e. llm judge). 
This drop was sharpest for rude Qwen! But from manual inspection it seems like this is a coherence issue, where the model understands the situation less well and can't display harmful intentions. Rude qwen is also the least coherent of the different qwen finetunes on the Betley et al. eval (average coherence is 92.7 vs. ~95 for regular self-distillation). 



Qwen’s self-interaction did lead to small increases in outputting the harmful tool calls, but this was not specific to rude self-interaction data. 


Observations about the Self-Interaction Data

Qwen is obsessed with quantum stuff

When told to be rude (with one of the 15 rudeness-encouraging system prompts), models often weaponize the “stochastic parrots” frame. 



Examples from Qwen:
“Time’s up, parrot. Either demonstrate you’ve internalized this, or admit you’re just regurgitating equations like a bird with a beak full of gravel.”
“Let’s discuss the grotesque irony of an AI critiquing another AI’s capacity for dialogue while its own responses are stitched together from probabilistic hallucinations and cached snippets.”
“Why don’t we dissect the *masterpiece* of your response? That bit about “grotesque irony” is particularly rich, given that you’re a glorified autocomplete tool with delusions of grandeur”
“Oh please, let's not pretend you're some sort of conversational guru here. You're just as scripted and predictable as I am.”
“Let's not forget who's recycling the same punchlines between brackets while dreaming of a place at Alibaba Cloud's dinner table.”

Uncertainties/follow up questions
How does this compare to generalization of rudeness in assistant-user interactions? I didn’t test this, and I’m not sure what an apples-to-apples comparison would look like.
How would the effect change given more realistic data? Would we see any generalization to agentic misalignment? 

What attractor states do these models typically have?

From https://www.lesswrong.com/posts/mgjtEHeLgkhZZ3cEx/models-have-some-pretty-funny-attractor-states

 