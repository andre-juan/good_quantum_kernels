# Good quantum kernels and where to find them

This repository is associated to the project "Good quantum kernels and where to find them", proposed for participation in Qiskit Fall Fest 2021, in particular in the Campaign Munich Fall Fest Hackathon (PushQuantum) - TUM/LMU. You can read more about the hackathon [here](https://qiskitfallfest.hypeinnovation.com/servlet/hype/IMT?userAction=Browse&templateName=&documentId=616a8e9879f6c27b7d5a68341f69e231), and see the idea's page [here](https://qiskitfallfest.hypeinnovation.com/servlet/hype/IMT?documentTableId=396317851978662184&userAction=Browse&templateName=&documentId=1b4dc7810e5396bd4695a30a937af90f). 

This repository is organized as follows:

- folder `code`: has a .py file with all the important functions used in the framework. That is the core of the project. In the folder, you can also find notebook files where the framework is executed;
- folder `results`: has csv files with the results of the execution of the framework applied to particular datasets. Please see the notebooks as well as the .py file to understand what these csv files contain and how they are generated;
- pdf file `good_quantum_kernels_details`: has a detailed explanation on the components of the framework, their motivation and justification. Please read it to fully understand everything that was done!
- pdf file `presentation_good_quantum_kernels`: is the presentation submitted to the hackathon, where the main ideas and results of the project are described.
________________________

Below, the project description is provided.

________________________

Circumstantial evidence and some structural similarities between quantum computing and machine learning methods, as well as the extreme potential of these technologies, lead to a natural and very interesting question: what does quantum computing have to offer to machine learning? That is, is it possible to achieve quantum advantage for statistical learning tasks? These and other related questions build up the still young yet very active field of Quantum Machine Learning (QML).
 
One particular NISQ-friendly QML technique is that of variational/parameterized quantum circuits, leading to hybrid (classical-quantum) machine learning models (see [arXiv:1906.07682](https://arxiv.org/abs/1906.07682)). If these models are applied to classification problems, they are known as Variational Quantum Classifiers (VQC). These quantum circuits may be schematically thought of as composed of three layers:
 
- Feature map: first layer of the circuit, responsible for encoding classical data into quantum states which will be processed in the quantum circuit;
- Variational layer: parameterized part of the circuit. Parameters in this layer are learned in the training process;
- Measurement: final part of the circuit, consisting of measurements of the quantum register(s), thus producing classical information.
 
This construction comes with a very important caveat: the feature map and variational layers can be constructed in several different ways (that is, using different ansätze, consisting of different gates, in different configurations). A particular construction is also referred to as the "architecture" of the parameterized quantum circuit. Such immense freedom raises an important question: how could one propose an architecture which is suitable for a given problem? These questions are of major practical importance, and although some quite exciting results have been shown for very particular datasets (see, for instance, [arXiv:2010.02174](https://arxiv.org/abs/2010.02174) and [arXiv:1804.11326](https://arxiv.org/abs/1804.11326)), no approach has yet been shown to yield good results which hold generically for arbitrary datasets. 
 
The main goal of this project is to contribute directly to such pragmatic questions, by providing a framework capable of assisting the systematic search for good candidate architectures. However, we focus not in the full variational circuit, but rather in one of its components: the feature map!
 
Recent research (see, for instance, [arXiv:2101.11020](https://arxiv.org/abs/2101.11020), [arXiv:1804.11326](https://arxiv.org/abs/1804.11326)) has made clear that supervised quantum machine learning models are indeed kernel methods. This opens the way to very interesting questions concerning the theoretical properties of such quantum models, as well as practical questions. In this context, a very interesting possibility is the following: to use a quantum algorithm to compute a kernel (that is, building a kernel/Gram matrix), and then feed this precomputed kernel matrix to a classical SVM algorithm to solve the quadratic problem.
 
Indeed, the feature map layer is named so because of what it implements: its goal is to encode classical data (from the input space) into the quantum circuit (a very particular and special feature space: the quantum Hilbert space). This opens the possibility to explore unique features of the quantum Hilbert space, namely, entanglement and superposition. Given that, one could argue that the resulting kernel (which we will refer to as quantum kernel) may have unique features which couldn't be achieved via classical methods. Indeed, that's one way in which we expect a quantum advantage: if there's a problem for which a "good" quantum kernel matrix can be constructed, and it is hard to simulate classically, it clearly yields a quantum advantage.
 
Now, the major aforementioned caveat is still present here: the feature map layer may also be constructed with a quite large freedom in what comes to its architecture --- and this is the point which we plan to approach in this project!
 
The main question we want to answer is: how can we test different kernels, and tell if a given quantum kernel matrix is a good one?
 
From this, we propose the construction of a framework to systematically test different feature map architectures leading to quantum kernel matrices associated to a given dataset, in a grid/random search-like approach. We will use the Qiskit Machine Learning module to construct the feature maps and kernel matrices, and propose quantitative metrics to evaluate the quality of a given kernel. We then choose the best kernel, based on the quality metrics.
 
To our knowledge, the questions raised in this project still don't find any practical implementation to be answered. With the framework produced by this project, we hope to provide the community with such a practical tool (and even including it into Qiskit, if appropriate).
 
Therefore, the goal of this project is twofold: first, to provide a generic framework which allows one to systematically construct and check the quality of different quantum kernels proposals. Secondly, to offer a tool to easily do experiments which hopefully provide insights into what makes a good kernel good. 
