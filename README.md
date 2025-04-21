# HW-for-AI-ML

This repository for Brandon Hippe's work for ECE 510 (Hardware for AI and ML)

## Heilmeier Questions
**1. What are you trying to do? Articulate your objectives using absolutely no jargon.**

I will be trying to implement a neural network to solve [graph coloring](https://en.wikipedia.org/wiki/Graph_coloring) problems. These are problems where neighboring elements (nodes) of a mathematical graph must belong to different sets. Some famous examples of applications of these problems are the [four color map theorem](https://en.wikipedia.org/wiki/Four_color_theorem) and Sudoku puzzles.

**2. How is it done today, and what are the limits of current practice?**

Graph coloring problems are classified as [NP-Complete](https://en.wikipedia.org/wiki/P_versus_NP_problem#NP-completeness). This means that while a graph coloring can be verified as a solution in polynomial time, it is unlikely to be possible to generate a solution in polynomial time (that is, unless P=NP).

**3. What is new in your approach and why do you think it will be successful?**

I will be aiming to utilize a hardware-implemented neural network implementation to accelerate the process of generating potential solutions to the graph coloring problem. In many applications of graph coloring problems, the best-case solution (e.g a minimally colored graph) is not necessary, rather any solution at all will work. As verifying a solution can be performed in polynomial time, it is likely possible to keep verification of the given solution performed in software.

**4. Who cares? If you are successful, what difference will it make?**

Graph coloring problems are useful in scheduling problems and register allocation, both of which occur frequently in modern computer architectures. Generating solutions to these problems quickly would be useful in improving hardware utilization, especially if the solutions are close to minimal colorings.

**5. What are the risks?**

It is certainly possible that a neural network trained to generate graph colorings would only be able to color very specific types of graphs. Furthermore, without some kind of randomized initialization/selection process, an accelerator like this is worthless if the only solution it can generate is incorrect.

**6. How much will it cost?**

The cost of this implementation only comes at the expense of time over the term. All tools necessary to complete are either freely available or already obtained.

**7. How long will it take?**

This project is intended to take place over one term (10 weeks, less the initial introduction). This will include creating a model algorithm, profiling, translation into RTL, and benchmarking results.

**8. What are the mid-term and final “exams” to check for success?**

The midterm exam is a software neural network model capable of generating graph colorings. I see this as successful if the model is mostly (>50% of the time) capable of finding a valid coloring within a reasonable number of tries (not sure what this number is, likely <1000?). The final exam is a co-designed accelerator of this model to be compared with the software implementation, with a successful implementation matching the outputs of the software model and capable of decreased generation time.
