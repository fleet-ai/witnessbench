"""Training of a model that plays the games through the harness (`witness_agent`): multi-turn GRPO on the miles
trainer, where every reflection call of an episode gets the episode's one advantage ("broadcast" advantages).

`rewards` gives every reflection call of an episode its reward, `advantages` turns episode rewards into one
advantage per episode, `rollout` plays one training episode against the inference server, and `hooks` holds the
two entry points miles calls. Only `hooks` imports miles.
"""
