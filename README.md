# Graph Coloring Accelerator

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

## Codefest Presentation

![Codefest Slide](./codefest_slide.svg)
<img src="./codefest_slide.svg">

## Final Project State

Unfortunately, this project was unable to reach the point of developing/testing any hardware acceleration implementations for this algorithm. While there are examples of using neural network implementations to solve sudoku puzzles, I was unable to duplicate any of these results with my intended architecture in mind. In retrospect, I believe there are a number of reasons for this.

First and foremost, many of the existing solver networks implement a one-at-a-time approach, where the network outputs the probability of each possible assignment for each node in the graph. This is used to update the graph with the most likely assignment across all nodes, with the updated graph repeatedly fed back into the network to fully color the graph. While this does provide a speed improvement over the brute force approach I implemented here to compare to, it comes at the cost of being a "greedy" algorithm, which means it is not guaranteed to find the solution, and would therefore oftentimes require multiple attempts to find the solution.

Second, it occurs to me now that Sudoku puzzles are not the best choice for training a generalized graph coloring network on. A "good" sudoku puzzle is sufficiently constrained such that there is one and only one possible solution (or coloring). This runs contrary to many of the more useful applications of graph coloring, where the conditions are oftentimes far less constrained and there may exist huge numbers of valid colorings, and the more useful question to answer is not "can I generate a coloring?" but "can I generate a 'minimal' coloring?", for some sense of the word minimal (e.g fewest time needed to complete tasks among many parallel compute units, or fewest compute resources used, etc.). My approach to generating colorings was to generate the full coloring all at once, which I believe is a valid approach for more 'sparse' graphs where some kind of optimization is desired, but has a very low chance of generating any valid sudoku solution, let alone the one for the given puzzle. For context, a 9x9 grid with 9 possible colors per cell has approx. ~$1.97\cdot10^{77}$ combinations, with approx. ~$6.67\cdot10^{21}$ of those being [valid sudoku solutions](https://en.wikipedia.org/wiki/Mathematics_of_Sudoku) (though only ~5.5 billion of those are unique w.r.t symmetry). I did notice many of my attempts to use an LLM (GPT 4.1, in my case) to write python code to implement a NN sudoku solver used the one-at-a-time approach, though I always attempted to modify those to generate the full coloring in a single shot, and otherwise, I prompted it to make generalized coloring networks, which worked great on the smaller examples I tested.

In hindsight, I view my approach to this as another misguided attempt to create a generalized solution for a highly-specific problem, which limits the applicability of these co-design principles for creating hardware accelerators. In another attempt at this project, I would likely switch to attempting a smaller-scale approach with some optimization in mind, for example, functional unit allocation with maybe 10-20 nodes with a "minimal" coloring as a goal. I suspect this would be much more doable for the type of hardware acceleration I had in mind.