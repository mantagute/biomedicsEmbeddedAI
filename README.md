# Neural Networks on FPGAs — Study Notes

> An asynchronous study guide on embedded AI and neural network deployment on FPGAs.

---

## Why Not Just Use a GPU?

When you train a neural network, you typically do it on a GPU. GPUs have large memory banks, massive parallelism, and native support for floating-point arithmetic — everything a training workload needs.

FPGAs are a completely different environment. You do not have the luxury of taking a full-blown model and dropping it onto the chip. Even very simple models require careful adaptation. **On an FPGA, everything is about optimization.**

That said, FPGAs offer real advantages for inference at the edge:

- Deterministic, ultra-low latency
- Low power consumption compared to GPUs
- Fully customizable parallelism in hardware
- No operating system dependency — the logic runs directly on silicon

### A Note on Parallelism

Parallelism means doing multiple things at exactly the same time — not switching between them quickly, but literally simultaneously.

A CPU executes instructions one after another (or a handful at a time with multiple cores). A GPU improves on this with thousands of smaller cores that can each run the same operation on different data simultaneously — which is why matrix multiplications in neural networks map so well to GPU hardware.

An FPGA takes a different approach entirely. Instead of running a program on general-purpose cores, you **build the computation directly into the chip's wiring**. Every operation you need has its own dedicated piece of hardware, and all of those pieces can run at the same time, every clock cycle.

Think of it this way:

```
CPU:   [multiply] → [multiply] → [multiply] → [multiply]   (sequential)
GPU:   [multiply]   [multiply]   [multiply]   [multiply]    (parallel, shared cores)
FPGA:  [multiply]   [multiply]   [multiply]   [multiply]    (parallel, dedicated circuits)
        ↑ neuron 1   ↑ neuron 2   ↑ neuron 3   ↑ neuron 4
```

On an FPGA, if your model has 64 neurons in a layer, you can instantiate 64 multipliers in hardware and compute all of them in a single clock cycle. There is no scheduling, no memory bus contention, no core sharing. The parallelism is structural — it exists in the physical layout of the chip.

---

## The Physical Anatomy of an FPGA

An FPGA is not a general-purpose processor. It is a chip filled with a large array of configurable building blocks, connected by a programmable routing fabric. Understanding what each block does — and why it exists — is essential for reasoning about what fits on a chip and what doesn't.

Here is an overview of the main physical resources you will encounter.

---

### LUT — Look-Up Table

The LUT is the fundamental logic building block of an FPGA. Every piece of combinational logic you can imagine — AND gates, OR gates, XOR gates, multiplexers, comparators — is implemented using LUTs.

A LUT with N inputs is essentially a small truth table stored in memory. It has 2ᴺ possible input combinations, and for each one, it stores a pre-computed output bit. When signals arrive at the inputs, the LUT looks up the corresponding output instantly.

```
4-input LUT — 2⁴ = 16 stored output bits

Inputs: A B C D
           ↓
   [ truth table in memory ]
           ↓
        Output: 1 bit
```

Because the truth table is programmable, a single LUT can implement any logic function of its inputs. A 4-input LUT can be an adder one day and a comparator the next — it depends entirely on what the bitstream writes into its memory cells.

LUTs are what get consumed when your HDL code describes logic. An adder uses LUTs. A multiplexer uses LUTs. A decoder uses LUTs. When a synthesis report says "you are using 70% of LUT resources," it means 70% of those truth-table blocks are occupied.

---

### FF — Flip-Flop

A Flip-Flop (also called a register) stores a single bit of state. It captures the value on its input at the moment of a clock edge and holds it steady until the next clock edge.

```
        clock edge
            ↓
Input ──► [ FF ] ──► Output (held stable until next clock)
```

LUTs compute things. Flip-Flops remember things — for exactly one clock cycle at a time.

In a neural network context, FFs are used to pipeline results between stages. If you have multiple layers of computation, each layer's output can be registered into FFs so the next layer can start processing while the previous one is already accepting new inputs. This is **pipelining**, and it is how FPGAs achieve high throughput alongside low latency.

LUTs and FFs almost always appear together. On most FPGAs, they are physically paired inside a structure called a **Slice** or **Logic Element** — one LUT and one or more FFs sharing the same physical cell. This pairing exists because combinational logic (LUT) and state storage (FF) are almost always used together.

---

### DSP Block — Digital Signal Processor

DSP blocks are hardened arithmetic units built directly into the silicon — unlike LUTs, which implement logic by lookup, DSP blocks contain actual multiplier and accumulator circuits etched into the chip.

A typical DSP block can perform:

```
result = (A × B) + C
```

in a single clock cycle, on operands of 18 to 27 bits depending on the chip family.

This matters enormously for neural networks. The core operation of every neuron is a **multiply-accumulate**: multiply each input by its weight, then sum the results. A DSP block does exactly this, far more efficiently than building the same operation out of LUTs.

```
Neuron output = Σ (input[i] × weight[i])
                       ↑
              This is what DSP blocks are built for
```

When you quantize your model to INT8 or INT16, the operands fit neatly into DSP block inputs, and you can pack more operations per block. When weights are `float32`, you either need to chain multiple DSPs together or fall back to LUTs — both of which are more expensive. This is one of the concrete reasons quantization matters on FPGAs.

---

### BRAM — Block RAM

BRAM is dedicated on-chip memory. Unlike external DRAM (which is off-chip, slow, and energy-hungry), BRAM sits directly on the FPGA die and can be read or written in a single clock cycle.

Each BRAM block is a small, dual-port memory — typically 18 Kb or 36 Kb in size. Dual-port means it has two independent access ports, so two different parts of your circuit can read from or write to it simultaneously.

In a neural network on an FPGA, BRAM is used to store:

- **Weights** — the parameters of the network, loaded from the bitstream or from external memory at startup
- **Activations** — intermediate values between layers, especially in pipelined designs
- **Input buffers** — staging area for incoming data before it enters the computation

The critical constraint is that BRAM is scarce. A mid-range FPGA might have a few megabytes of total BRAM. A neural network with millions of parameters simply will not fit. This is precisely why quantization (smaller weights) and pruning (fewer weights) are not optional — they are the difference between a model that fits and one that does not.

```
float32 weight × 1M parameters = 4 MB  →  likely does not fit
int8 weight    × 1M parameters = 1 MB  →  might fit
int8 weight    × 100K parameters (pruned) = 100 KB  →  fits comfortably
```

---

### Programmable Interconnect

The blocks above — LUTs, FFs, DSPs, BRAMs — are arranged in a fixed grid across the chip. What makes an FPGA reconfigurable is the **programmable interconnect**: a dense mesh of wires running horizontally and vertically between all resource blocks, with configurable switches at every intersection.

The bitstream programs these switches to either connect or disconnect, effectively routing signals between whatever blocks your design needs to communicate. This is the physical substrate that Place & Route operates on.

The interconnect is fast, but not free. Every switch a signal passes through adds a small delay. Long routes across the chip accumulate more delay than short, local routes. This is why the placement decisions made during Place & Route directly impact how fast your circuit can run.

---

### How They Work Together in a Neural Network

A single layer of a neural network on an FPGA uses all of these resources in concert:

```
BRAM          →  stores the layer's weights
DSP blocks    →  compute weight × input for each neuron
LUTs          →  implement the activation function (e.g. ReLU, comparisons)
Flip-Flops    →  pipeline the output to the next layer
Interconnect  →  carries signals between all of the above
```

The synthesis and place & route tools are responsible for mapping your high-level description onto this physical substrate, respecting the available count of each resource type. When a design fails to fit, it is usually because one resource — most commonly BRAM or DSP blocks — has been exhausted.

---

## The Core Challenge: Constraint vs. Capability

The fundamental tension in embedded AI is this: machine learning models are designed in environments with abundant resources, but deployed in environments where every bit of memory, every logic block, and every clock cycle is scarce.

This is what makes FPGA-based AI different from cloud inference or even microcontroller deployment. You are not running software on top of hardware — you are **turning the model itself into hardware**.

---

## Optimization Techniques

Before a neural network can run on an FPGA, it must be compressed and restructured. The two most important techniques are quantization and pruning.

### Quantization

Neural networks are typically trained using 32-bit floating-point numbers (`float32`). These are precise but expensive — in terms of memory, bandwidth, and the size of the arithmetic units required to compute them.

Quantization reduces this precision to smaller integer representations:

```
float32  →  4 bytes per weight
int8     →  1 byte per weight    (4× smaller)
int4     →  0.5 bytes per weight (8× smaller)
```

Fewer bits means:
- Smaller memory footprint
- Less silicon area on the FPGA
- Faster arithmetic operations in hardware

The accuracy loss from quantization, when done carefully, is minimal. The model learns to compensate during a quantization-aware training step.

### Pruning

Not all weights in a neural network contribute equally to its output. Many weights end up close to zero — they could be removed without meaningfully affecting predictions.

Pruning identifies and removes these near-zero weights, producing a **sparse network**:

- Fewer multiplications to perform
- Zero-weight connections can be eliminated entirely in hardware
- The resulting model is smaller and faster without retraining from scratch

Pruning and quantization are often applied together for maximum compression.

---

## From Model to Hardware: The HDL Pipeline

Once the model is optimized, it needs to be compiled into something an FPGA can actually run. This process goes through several stages.

```
HDL Code (VHDL / Verilog)
        │   loop unrolling applied
        ▼
High-Level Synthesis
        │   converts logic description into circuit netlist
        ▼
Place & Route
        │   maps netlist onto physical LUTs, FFs, and DSPs
        ▼
Bitstream  (.bit / .rbf / .sof)
        │   compiled binary — encodes every switch and connection
        ▼
Programmed onto the FPGA
```

### HDL and Loop Unrolling

Hardware Description Languages like VHDL and Verilog describe digital circuits, not sequential programs. One key technique when mapping neural network computations to HDL is **loop unrolling**: instead of executing a loop iteration by iteration, each iteration becomes independent parallel hardware.

```
// Sequential (software)
for i in range(4):
    sum += weight[i] * input[i]

// Unrolled (hardware — all computed simultaneously)
sum = w0*x0 + w1*x1 + w2*x2 + w3*x3
```

This is what gives FPGAs their latency advantage: operations that would take N clock cycles sequentially can happen in a single cycle when unrolled.

### Synthesis

High-level synthesis tools translate the HDL description into a **circuit netlist** — a representation of logic gates and their connections. This step applies optimizations like resource sharing, pipelining, and timing analysis.

### Place & Route

Place & Route is the step where an abstract circuit description becomes a physical layout on a specific chip. It is one of the most computationally intensive steps in the FPGA toolchain, and understanding it explains a lot about why FPGA compilation takes minutes or hours rather than seconds.

**What the netlist contains at this point**

After synthesis, you have a netlist: a list of logical components (a multiplier here, an adder there, a register somewhere else) and the connections between them. It describes *what* the circuit does, but not *where* it lives on the chip.

**Placement**

The FPGA die contains thousands to millions of physical resource blocks arranged in a fixed grid — LUTs, Flip-Flops, DSPs, BRAMs. Placement is the process of deciding which logical component from the netlist gets assigned to which physical block on that grid.

This matters because the chip is finite and its blocks are fixed in location. A multiplier from your neural network layer must be mapped to a DSP block that actually exists on the chip, in a real physical position.

```
Netlist (abstract)          FPGA die (physical grid)
─────────────────           ──────────────────────────────
Multiplier A         →      DSP block at column 4, row 12
Adder B              →      LUT cluster at column 5, row 12
Register C           →      Flip-Flop at column 5, row 13
```

A good placement groups elements that communicate frequently close together on the die, minimizing the distance signals must travel.

**Routing**

Once every element has a position, routing connects them using the FPGA's programmable interconnect — a grid of wires and configurable switches that run between the resource blocks.

This is where the tool must literally find physical wire paths through the chip for every connection in your netlist. Two elements might need to be connected but the most direct wire path is already occupied by another signal. The router has to find an alternative path, which may be longer and therefore slower.

**Why wire length matters: timing**

Signals in digital circuits travel fast, but not instantaneously. A longer wire introduces more propagation delay. If the path between two elements is too long, the signal might not arrive before the next clock edge — and the circuit produces wrong results.

The place & route tool runs **timing analysis** continuously, checking whether every connection meets the required timing constraints. If something fails, it tries a different placement or routing path and checks again. This iterative process is why it takes so long.

```
Short path  →  fast signal  →  timing met   ✓
Long path   →  slow signal  →  timing fail  ✗  → tool reroutes and retries
```

**The output: a fully specified physical layout**

When place & route succeeds, every logical element has a physical location, every connection has a physical wire path, and every timing constraint has been verified. This complete specification is what gets encoded into the bitstream.

### Bitstream

The final output is a **bitstream** — a binary file that encodes the complete configuration of the FPGA: every LUT function, every connection, every register. Loading this file onto the chip transforms it into a dedicated inference accelerator.

| Extension | Vendor              |
|-----------|---------------------|
| `.bit`    | Xilinx / AMD        |
| `.rbf`    | Intel / Altera      |
| `.sof`    | Intel (JTAG target) |

---

## Inference on an FPGA

Inference is the AI doing its actual work — taking new inputs and producing predictions. This is fundamentally different from the training phase.

A few important distinctions from the LLM world:

- The model is **fixed at compile time** — weights are baked into the bitstream
- There is no memory of previous inputs between runs
- Latency is measured in **microseconds**, not seconds
- No GPU, no operating system, no runtime interpreter — just logic on silicon

In practice, inference on an FPGA looks like feeding data into a hardware IP core and reading back a result. The neural network has become a circuit.

### Training vs. Test Data

Even in the embedded context, the principle of separating data applies:

- **Training data** is used to learn the model's weights
- **Test data** is held back and only used to evaluate final accuracy

This separation is especially important when validating a quantized or pruned model, since compression can introduce small accuracy losses that only become visible on unseen data.

---

## Key Concepts at a Glance

| Concept | What it means |
|---------|---------------|
| **FPGA** | Reconfigurable chip — logic is defined by a bitstream, not fixed in silicon |
| **Bitstream** | Binary file that programs every switch and connection inside the FPGA |
| **LUT** | Look-Up Table — the basic logic building block of an FPGA |
| **DSP Block** | Dedicated multiplier/accumulator unit, critical for neural network math |
| **BRAM** | On-chip memory — limited, so weight storage must be carefully managed |
| **HDL** | Hardware Description Language (VHDL, Verilog) — describes circuits |
| **HLS** | High-Level Synthesis — compiles C/C++ or model descriptions into HDL |
| **Quantization** | Reducing weight precision from float to integer |
| **Pruning** | Removing near-zero weights from a trained network |
| **Loop Unrolling** | Expanding loops into parallel hardware operations |
| **Inference** | Running a trained model on new data to produce predictions |

---

## Further Reading

- [An Introduction to FPGAs — SparkFun](https://learn.sparkfun.com/tutorials/programmable-logic-introduction)
- [Efficient Deep Learning — MIT HAN Lab](https://efficientdeeplearning.mit.edu/)
- [FPGA vs GPU for Inference — IEEE Spectrum](https://spectrum.ieee.org)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference — Google](https://arxiv.org/abs/1712.05877)