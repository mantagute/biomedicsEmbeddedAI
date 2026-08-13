# Machine Learning and Deep Learning — from absolute zero, with the technical terms built in

This text assumes you've never heard of "training a model" before. Every idea is born
from a concrete problem, and as soon as the idea appears, the corresponding technical
term is introduced — so you never have to memorize a word without knowing where it
comes from.

---

## First: what a computer program normally is

A regular program works like this: a human writes exact rules, and the computer just
obeys. "If the temperature is above 30 degrees, show 'it's hot'" — you decided that
rule, the computer just followed it.

This works well when the rule is clear. The problem shows up when the rule is
impossible to write by hand.

---

## The problem hand-written rules can't solve

Imagine you want the computer to look at a photo and say "this is a cat" or not.
Try writing that rule. "Triangular ear" — dogs have those too. "Whiskers" — might not
show up in the photo. No matter how many rules you write, there's always an exception
left over. The pattern that separates "cat" from "not-cat" is too messy to turn into a
list of exact instructions.

**This is exactly the kind of problem Machine Learning exists to solve.**

---

## The turning point: instead of writing the rule, you show examples

Instead of you writing the rule, you show many already-solved examples and let the
computer figure out the rule itself.

You gather a million photos, and for each one you write "cat" or "not-cat" — this is
the **label**, the answer key for that example. When each example comes with an
answer key, this type of learning is called **supervised learning** — it's the
supervisor (the answer key) guiding the process. When there's no answer key at all,
and the machine just tries to find patterns and groupings on its own, this is called
**unsupervised learning**.

The process of "the algorithm gradually adjusting itself by looking at labeled
examples" is called **training**. The set of examples used for this is the
**training set**.

---

## What this "machine of knobs" literally is

Behind all this there's a mathematical function — think of it as a machine with
knobs. You feed in an input (the photo), the machine does internal calculations, and
spits out an output (the prediction: "cat" or "not-cat"). This whole machine is what's
called a **model**. The adjustable knobs inside it are the **parameters**, also
called **weights**.

At the start, the weights are in random positions, so the model guesses randomly.
**Training is the process of turning these knobs, little by little, until the model
gets things right more often.**

After training, when you use the model on a new photo (outside the training set) to
get a prediction, this is called **inference** — the "usage" phase, as opposed to the
"learning" phase.

---

## How the model knows it was wrong, and by how much

To adjust the weights, the model needs a way to measure "how wrong" the guess was.
This error-measuring device is the **loss** (loss function, also called the **cost
function**). It's a number: the higher it is, the worse the prediction; the closer to
zero, the better.

If the prediction is a continuous number (e.g., predicting tomorrow's temperature),
this type of problem is called **regression**, and the typical loss is called **mean
squared error**, or **MSE** — it squares the error, punishing large errors more than
small ones.

If the prediction is a category (e.g., "cat" or "not-cat"), the problem is called
**classification**, and the typical loss is called **cross-entropy** — it punishes
more heavily when the model is very confident and wrong.

Worth noting: the loss guides training, but it isn't the same thing as how you
**evaluate** the model once it's ready. For evaluation, you use **metrics** like
**accuracy** (% correct), **precision** and **recall** (how much the model errs in one
direction or another), and the **confusion matrix** (a table showing exactly where it
confuses one class with another). The loss is the "thermometer during training"; the
metric is the "final report card," meant for humans to understand.

---

## How the model decides which direction to turn each knob

There are thousands of weights. Which one do you turn, in which direction, and by how
much? The answer comes from an idea called the **gradient**: for each individual
weight, the gradient tells you "if you turn this weight a little bit, does the error
go up or down? And by how much?"

Picture a dark room, trying to walk down a hill. You can't see the whole path, but you
can feel with your feet whether the ground slopes down or up right where you are. You
take a step in the downward direction, feel again, repeat. This process — feeling the
slope (gradient) and taking a step in the opposite direction — is called **gradient
descent**.

- The size of each step is the **learning rate**. Too big a step, and you might
  "jump over" the bottom of the valley and end up bouncing back and forth (the loss
  oscillates or even gets worse). Too small a step, and you'll get there, but too
  slowly.

- Calculating this gradient in a model with many layers (which we'll see next)
  requires reusing calculations from back to front, layer by layer — this specific
  calculation algorithm is called **backpropagation**. It's "just" the engineering of
  computing the gradient in a feasible way for a large network.

- Taking each step based on *all* the examples at once would be correct, but
  extremely slow and expensive. In practice, each step is taken by looking at only a
  small chunk of the data at a time — a **batch**. This makes the gradient a noisy
  but fast-to-compute estimate, and this method has a name: **stochastic gradient
  descent**, or **SGD**.

- Each weight update using a batch is an **iteration**. One complete pass through all
  the batches — that is, through the whole dataset once — is an **epoch**. Training
  normally means repeating dozens or hundreds of epochs.

---

## "Pure" gradient descent has a problem — that's where optimizers come from

Descending by only feeling the current slope is naive on difficult terrain: if the
valley is narrow and elongated, the plain gradient ends up zigzagging back and forth
instead of going straight. To fix this, there are smarter versions of gradient
descent, called **optimizers**:

- **Momentum**: gives the movement "inertia" — as if you had mass and didn't change
  direction abruptly. This reduces the zigzagging.
- **Adam** (*Adaptive Moment Estimation*): the most widely used optimizer today. It
  automatically adjusts the step size for each weight individually, combining the
  idea of momentum with scale adaptation. In practice, it works well without much
  manual tuning.

The algorithm that actually applies "new weight = old weight − step in the direction
of the gradient" is the optimizer — plain gradient descent is just its simplest case.

---

## The full training cycle

```
1. show a batch of examples to the model
2. the model guesses an answer (forward pass)
3. compare the guess with the answer key → compute the error (loss)
4. compute the gradient for each weight (backpropagation)
5. the optimizer uses that gradient to adjust the weights
6. repeat with the next batch, until an epoch is complete
7. repeat for several epochs
```

`forward pass` is the technical name for "passing the input through the machine and
getting the prediction" — as opposed to the `backward pass`, which is the gradient
calculation going from back to front.

---

## Why the model needs to see examples that never appeared during training

If you only train with photos of black cats, the model might learn "cat = black
thing" instead of the actual shape of a cat — it would do great on the examples it
saw but get an orange cat wrong. This risk of "memorizing instead of learning the
general pattern" is called **overfitting**. The opposite — a model so simple or
poorly trained that it doesn't even learn its own training examples — is called
**underfitting**.

There's even a name for this tension between the two: the **bias-variance
tradeoff**. A model that's too rigid has **high bias** (can't capture the real
pattern → underfitting). A model that's too flexible has **high variance** (captures
even the specific noise of the examples → overfitting). You never zero out both at
the same time, you just look for the balance.

To catch this, examples are always split into three groups:

- **Training set**: what the model actually uses to adjust its weights.
- **Validation set**: used during training, without directly adjusting weights on
  it, just to track "is the model generalizing, or just memorizing?"
- **Test set**: used only once, at the end, as an honest and definitive proof of
  performance.

There are specific techniques, called **regularization**, created just to fight
overfitting, forcing the model to not rely too heavily on specific details of the
training data:

- **Dropout**: randomly turns off neurons during training, preventing knowledge from
  becoming concentrated in just a few pathways.
- **Weight decay** (or L2 regularization): penalizes very large weights inside the
  loss itself, encouraging smaller, more "restrained" weights.
- **Data augmentation**: generates artificial variations of the training examples
  (rotating, mirroring, changing the brightness of a photo, for example), increasing
  diversity without needing to collect more real data.
- **Early stopping**: stops training at the moment the validation loss stops
  improving, even if the training loss keeps dropping — a sign that from that point
  on the model is just memorizing.

---

## Why hyperparameters exist, separate from parameters

Weights are adjusted automatically by the gradient. But things like "which learning
rate to use," "what batch size," "how many layers" — these decisions aren't part of
the gradient's calculation; they're structural choices made by you, **before**
training starts. They're called **hyperparameters**, to distinguish them from
parameters (weights), which are adjusted from within the training process itself.

Tuning hyperparameters is normally trial and observation: you train, look at the
validation loss, change a hyperparameter, train again.

---

## Why data needs to be normalized before training

If one input feature ranges from 0 to 1 and another ranges from 0 to 10,000, the
"terrain" of the loss (remember the hill in the dark?) ends up extremely stretched in
one direction and squeezed in the other — descending it becomes unstable and slow.
**Normalization** is rescaling all inputs to similar ranges (for example, all
between 0 and 1, or all with mean 0), making the terrain more "rounded" and easier to
descend.

---

## What changes in Deep Learning: the machine becomes stacked layers

Everything up to this point applies to Machine Learning in general. What's different
about **Deep Learning** lies in the internal structure of the machine of knobs:
instead of a single simple machine, it's organized into **stacked layers**, forming a
**neural network**. Each layer takes the result of the previous one, transforms it a
bit more, and passes it along.

This allows the model to learn **increasing levels of abstraction**: in a photo, the
first layer might learn to notice simple edges; the next combines edges into shapes
(eye, ear); the next combines shapes into "this looks like a cat's face." Nobody
programs this by hand — it emerges from the weight adjustment itself, layer by layer.
"Deep" just means that: many stacked layers.

However, simply stacking purely linear layers (with nothing else) doesn't help:
mathematically, a composition of linear transformations is still equivalent to a
**single** linear layer — the depth would just be theater. That's why, between
layers, there's an **activation function** — a small non-linear transformation
(the most common ones are called **ReLU**, **sigmoid**, **softmax**) applied after
each layer. It's this non-linearity that gives the network the real ability to learn
complex patterns, curves, and sophisticated decision boundaries — without it, there
would be no real deep learning, just expensive, slow linear regression disguised as a
large network.

---

## Summary, with the technical term next to each idea

| Pedagogical idea | Technical term |
|---|---|
| Machine that turns input into a prediction | Model |
| Adjustable knobs inside the machine | Parameters / weights |
| Adjusting the knobs using examples | Training |
| Using the finished model on new data | Inference |
| Example with an answer key | Label |
| Learning with an answer key | Supervised learning |
| Learning without an answer key | Unsupervised learning |
| Measure of "how wrong" | Loss / cost function |
| Predicting a continuous number | Regression |
| Predicting a category | Classification |
| Final performance report card | Metric (accuracy, precision, recall, F1) |
| Feeling the slope of the error | Gradient |
| Descending opposite to the gradient | Gradient descent |
| Size of each step | Learning rate |
| Computing the gradient in a large network | Backpropagation |
| Chunk of data used per step | Batch |
| One weight update | Iteration |
| One full pass through the data | Epoch |
| Algorithm that applies the weight update | Optimizer (SGD, Momentum, Adam) |
| Memorizing instead of learning the pattern | Overfitting |
| Model that's too simple/weak | Underfitting |
| Balance between the two | Bias-variance tradeoff |
| Data set aside to check learning | Validation and test sets |
| Techniques to avoid rote memorization | Regularization (dropout, weight decay, data augmentation, early stopping) |
| Structural decisions before training | Hyperparameters |
| Making data scales similar | Normalization |
| Machine in stacked layers | Neural network / Deep Learning |
| Break from linearity between layers | Activation function (ReLU, sigmoid, softmax) |

---
