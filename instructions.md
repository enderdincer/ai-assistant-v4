1- create a uv python project.

2- this will be a multi-threaded application so we need a good framework for managing threads and their lifecycles. we should logs from threads too.

3- Use best practices like build interfaces first to define APIs and depend on the interfaces instead of custom implementations.

4- The application will be an ai assistant where there will be different inputs that come from different threads. This should be dynamic. I should be able to add multiple inputs dynamically of different and of the same type of inputs. An example is cameras. I will connect cameras and each stream will be processed by a thread then the thread will produce events. Another example of input type is text or audio.

5- These events will go into a priority queue (as I mentioned build a pub sub interface as I can use Kafka as an implementation later not an in-memory queue. but for now stick with the in-memory queue implementation and interface that is flexible for other things)

6- this whole thing is how the perception module works. call it perception folder and put everything here. there will be other modules like attention etc. but start with this module.
