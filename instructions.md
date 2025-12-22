1- now we will add real sensory processor to the perception system. an audio one.

2- a single input should accept many processors. For example, I should be able to add a speech to text model to produce text events from audio then I might add another processor to the same audio input to detect high frequency sound detection or maybe other audio pattern detection processor.

3- in this feature we are only going to the add an STT processor but adding sensory processors to inputs should be flexible.

4- details about the STT, this is the model we're going to use: https://huggingface.co/nvidia/canary-qwen-2.5b. it's simple. by using this model audio comes as input and transcribed text is produced as an event to the bus.

5- to make it flexible you can make the subscribe function accept a list of sensory processors.
