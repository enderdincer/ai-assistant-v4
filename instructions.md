# Refactor into Separate Components

1- The applications is a single component at the moment but it needs to be refactored into smaller units which is explained below.

2- The general structure of the components will be usualy "do one thing and share result with mqtt" or "listen mqtt and do only one thing"

3- The first standalone component is the audio collection service. it only pushes chunks/samples (whatever the audio unit is) to the corresponding mqtt topics

4- The second component is the transcription service. This one listens to the raw audio sources and constantly produces events to audio_transcribed topic with their source. it is able to transcribe multiple audio sources because raw audio topics include the audio source (name of the machine they run on).

5- Another service is the speech service. it only listens to a mqtt topic all/actions/speech and the message published here has the text as well as the voice (am_santa, bf_emma, etc.) and the speech speed (default 1.0)

6- There will be memory module as well. This will listen topics to store conversation history and facts.

7- Extraction service: This will listen conversations to extract facts or make psychological inferences and publish result to a mqtt topic

8- a text interaction service where there's no audio/microphone. it sends messages to assistant and listens for responses (all again mqtt)

9- and finally the assistant service this is where the brain lives. for now we will keep it very simple and it will only do request response.

## Some Notes

This is the refactor plan. Before starting make a comprehensive plan which includes the mqtt topics and hierarchy and how all components communicate with each other. make a diagram too, maybe mermaid.
