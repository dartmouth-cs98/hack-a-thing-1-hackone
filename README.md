# Hack-a-thing 1
## Will Sanford and Alex Quill
### 09/22/2020

### What we attempted to build
We both wanted to get more familiar with designing, building and training nearual networks, specifically with TensorFlow.
More specifically this is what each of use were looking to build with this. 
- Alex 
I have no experience with TensorFlow, and as such wanted to dive into as wide a scope as possible for hack-a-thing1. More specifically, I wanted to learn about two common problems in deep learning, **classification** and **prediction**. Two pretty paradigm examples of these types of problems involve the MNIST dataset and any of multiple large built-in tensorflow datasets. As such, I started off my coding by going searching for a TensorFlow MNIST image classification tutorial, and jumped to a variety of adjacent tutorials from there. TensorFlow.org contains a wide variety of TF tutorials that attack lots of different problems at varying skill levels. They are well-documented and contain copious links to documentation, more tutorials, and useful libraries. 

If I had more time I would move to building my own custom model using keras model subclassing. I appreciated how straightforward the 6 tutorials I did were, but I find that most of my learning at Dartmouth has come from trying, and failing, on my own. A lot of my interests with machine learning lie in biological prediction and classification problems, so I would be interested in training a custom model to predict, say, cardiovascular diagnoses based on hospital patient data. 

- Will : 
I have some experience with TensorFlow, but have never formally learned the basics. What I wanted to gain from this hack-a-thing was a formal understanding of the basics of the API, and then some time to explore a topic in two using the API that I had not had the proper training to explore in the past. I started by watching some YouTube tutorials and reading a textbook that a friend lent me. The combination filled in a lot of gaps in my knowledge that I had been missing. I then spent a small amount of time doing a few jupyter notebook tutorials in regression, classification and series forecasting as well as a small excersize with data I found myself.

The second part of what I did was explore ideas I have had in the past, and built a model to fit the idea. I had previously read many papers regarding how NNs are built to detect fake videos (deepfakes specifically). I went back to many of them, and tried to build my own model with certain aspects of many of them. I didnt have time to write a custom training loop, or find any real data to train it. The model uses an InceptionV3 CNN to extract feature vectors from frames of a video. These features are then fed into an RNN, which can be an LSTM or GRU depending on parameters passed to the model, that creates a variable amount of outputs. In the case of fake video detection, this output is a percentage. I have this style output as the default.

In the future (possibly another hack-a-thing) I want to actually train this model to predict some feature of a video and try to apply it to a live video feed.



### Who did what
- Alex I spent my time approximately as follows: ~2 hours of video content from the following tutorial, presented by Google Developer Josh Gordon at MIT's Center for Brains, Minds, and Machines (https://cbmm.mit.edu/video/getting-started-tensorflow-20-tutorial). This tutorial links to ~3 hours of Jupyter Notebook Tensorflow coding problems and walkthroughs, which I completed and placed in the "Tensorflow_Tutorials" folder. I used these tutorials to in turn explore 2-3 more tutorials, which cumulatively took up another ~2 hours. I integrated a **significant** amount of documentation exploration into each tutorial I completed, which took about ~3 more hours. I wanted to understand the code I was writing, and found I needed a solid brush-up on my understanding of CNNs, Backpropagation, and linear algebra to engage meaningfully with the tutorials' description of core deep learning concepts. 
- Will : I spent my time approximetly as follows: ~3 hours of YouTube tutorials, ~3 hours working through 'Hands-on Machine Learning with Scikit-Learn, Keras & Tensorflow' by Aurelien Geron that I borrowed from a friend.
~ 1.5 hours on tutorials and trying my own regression on healthcare data from (I only added the Housing Prices regression tutorial as all the tutorials were essentially the same). ~1 hour doing research about the model I wanted to build and how to make it effective and efficient. ~2.5 hours trying to build this model.

### What you learned 
- Alex: My learning fell into three main categories, each building on top of each other towards the actual code I wrote. First, I re-learned a lot of fundamental DL concepts from COSC 16, 'Computational Neuroscience.' I took this class 18F, and found that many of the core concepts had stuck around in my brain until this week - they just needed some refreshing. Building on these basics, I learned about Tensorflow and how it models neural networks behind the scenes using sequential models, tensors, keras and other libraries, etc. A lot of Tensorflow's wonderful simplicity is a product of a very complicated infrastructure, and I needed to understand a bit about what was going on under the hood before I moved to actually coding. My final learning category was that of how to actually use Tensorflow to build models. Besides the walkthroughs I completed on classification and prediction problems, I added on to my existing knowledge of numpy and matplotlib. 
- Will : I spent a lot of my time trying to learn how to build custom models (outside of Keras Sequential), which I think you 
would need to be able to do if you wanted to anything past one of the tutorials we worked on. I learned a lot about how to make these
from scratch and build your own custom training loops. The bulk of the knowledge I gained was syntactical as I think I have a fairly solid understanding of the underlying concepts in neaural nets, but there were certainly things that I needed to bulk up on regarding CNNs and RNNs specifically.

### What didn't work
- Alex: I need to write some more code on my own. If you dig into the jupyter notebooks I wrote, there's a fair bit of "playing around," which was undoubtedly useful for my learning, but very little organic, from-scratch code. I think the 10 hours I spent this week were a necessary baseline for 1) cleaning my own data and 2) building custom models, but I can't help but feel as if I'm turning in someone else's work, given that my code in this project is all tutorial-based. It's unsatisfying and feels dishonest. I'm looking forward to emulating Will's process a bit more and attacking a novel problem. 
- Will: For me, I think so much of using neural nets and a lot of machine learning is spent getting the right data and designing 
the proper network for your problem. When I tried to make something happen with healthcare data, the data turned out to be
nearly unusable for what I was trying to do. I think a lot of time needs to be spent on planning and researching before any code is actually 
written. I tried to do this a lot more in the model building excersize, which I think turned out way better, and I would actually be confident to move forward with it and use it in some real context.
